import json
from collections import defaultdict
from typing import Optional

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from backend.mcp.server.registry import mcp
from backend.mcp.server.models import AppContext
import logging

logger = logging.getLogger(__name__)


from config import DB_DESCRIPTIONS








# --------------------------------------------------
# Discovery tools
# --------------------------------------------------

@mcp.tool(name="get_available_databases", description="Get available databases with descriptions")
async def get_available_databases(ctx: Context[ServerSession, AppContext]) -> str:
    try:
        databases = [
            {"database": name, "description": DB_DESCRIPTIONS.get(name, "")}
            for name in ctx.request_context.lifespan_context.conn_ids.keys()
        ]
        return json.dumps(databases, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool(
    name="prefetch_databases",
    description="Open database connections eagerly for the provided database names.",
)
async def prefetch_databases(ctx: Context[ServerSession, AppContext], databaseNames: list[str]) -> str:
    try:
        opened = await ctx.request_context.lifespan_context.db.prefetch(databaseNames)
        return json.dumps({"opened": opened}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


















@mcp.tool(name="get_available_schemas", description="Get available schemas for a given database")
async def get_available_schemas(ctx: Context[ServerSession, AppContext], databaseName: str) -> str:
    try:
        conn_id = ctx.request_context.lifespan_context.conn_ids[databaseName]
        rows = await ctx.request_context.lifespan_context.db.query(
            conn_id,
            """
            SELECT schema_name FROM information_schema.schemata
            WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
            ORDER BY schema_name;
            """,
        )
        return json.dumps({"databaseName": databaseName,
                           "schemas": [row["schema_name"] for row in rows]}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)















@mcp.tool(name="get_available_tables", description="Get available tables for a given schema in a given database")
async def get_available_tables(ctx: Context[ServerSession, AppContext], databaseName: str, schemaName: str) -> str:
    try:
        conn_id = ctx.request_context.lifespan_context.conn_ids[databaseName]
        rows = await ctx.request_context.lifespan_context.db.query(
            conn_id,
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_type = 'BASE TABLE' AND table_schema = $1
            ORDER BY table_name;
            """,
            (schemaName,),
        )
        return json.dumps({"databaseName": databaseName, "schemaName": schemaName,
                           "tables": [row["table_name"] for row in rows]}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)















@mcp.tool(name="get_table_schema", description="Get column names and types for a given table in a given schema")
async def get_table_schema(ctx: Context[ServerSession, AppContext], databaseName: str,
                           schemaName: str, table_name: str) -> str:
    try:
        conn_id = ctx.request_context.lifespan_context.conn_ids[databaseName]
        rows = await ctx.request_context.lifespan_context.db.query(
            conn_id,
            """
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position;
            """,
            (schemaName, table_name),
        )
        if not rows:
            return json.dumps({"databaseName": databaseName, "schemaName": schemaName,
                               "table_name": table_name, "columns": [],
                               "error": f"Table '{schemaName}.{table_name}' not found."}, ensure_ascii=False)
        return json.dumps({"databaseName": databaseName, "schemaName": schemaName,
                           "table_name": table_name,
                           "columns": [{"column_name": r["column_name"], "data_type": r["data_type"]}
                                       for r in rows]}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


















@mcp.tool(name="get_database_catalog",
          description="Get lightweight database catalog with schemas, tables, geometry columns and SRID info.")
async def get_database_catalog(ctx: Context[ServerSession, AppContext],
                                databaseName: Optional[str] = None,
                                schemaName: Optional[str] = None) -> str:
    try:
        db_names = [databaseName] if databaseName else list(
            ctx.request_context.lifespan_context.conn_ids.keys()
        )
        catalog = []

        for db_name in db_names:
            conn_id = ctx.request_context.lifespan_context.conn_ids[db_name]
            params = []
            schema_filter_sql = ""
            if schemaName:
                schema_filter_sql = "AND t.table_schema = $1"
                params.append(schemaName)

            rows = await ctx.request_context.lifespan_context.db.query(
                conn_id,
                f"""
                WITH geom_info AS (
                    SELECT f_table_schema AS table_schema, f_table_name AS table_name,
                           f_geometry_column AS geometry_column, srid, type AS geometry_type
                    FROM geometry_columns
                )
                SELECT t.table_schema, t.table_name, g.geometry_column, g.srid, g.geometry_type
                FROM information_schema.tables t
                LEFT JOIN geom_info g
                  ON t.table_schema = g.table_schema AND t.table_name = g.table_name
                WHERE t.table_type = 'BASE TABLE'
                  AND t.table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
                  {schema_filter_sql}
                ORDER BY t.table_schema, t.table_name
                """,
                tuple(params) if params else None,
            )

            grouped = defaultdict(list)
            for row in rows:
                grouped[row["table_schema"]].append({
                    "table": row["table_name"],
                    "geometry_columns": (
                        [{"column_name": row["geometry_column"], "srid": row["srid"],
                          "geometry_type": row["geometry_type"]}]
                        if row.get("geometry_column") else []
                    ),
                    "recommended_geometry_column": row.get("geometry_column"),
                })

            catalog.append({
                "database": db_name,
                "schemas": [{"schema": sch, "tables": tbls} for sch, tbls in grouped.items()],
            })

        return json.dumps({"databases": catalog}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
















# --------------------------------------------------
# Validation + stats
# --------------------------------------------------

@mcp.tool(name="validate_sql_explain",
          description="Validate SQL syntax using EXPLAIN. Returns JSON with validated=true/false.")
async def validate_sql_explain(ctx: Context[ServerSession, AppContext],
                                sql: str, databaseName: str, step_number: int) -> str:
    try:
        sql_clean = (sql or "").strip()
        sql_upper = sql_clean.upper()

        if not sql_clean:
            return json.dumps({"validated": False, "step_number": step_number,
                               "database": databaseName, "error": "Empty SQL."})

        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return json.dumps({"validated": False, "step_number": step_number,
                               "database": databaseName,
                               "error": "Only SELECT or WITH statements are allowed."})

        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"]
        if any(word in sql_upper for word in forbidden):
            return json.dumps({"validated": False, "step_number": step_number,
                               "database": databaseName,
                               "error": "Forbidden SQL statement detected."})

        conID = ctx.request_context.lifespan_context.conn_ids[databaseName]
        await ctx.request_context.lifespan_context.db.query(conID, f"EXPLAIN (FORMAT JSON) {sql_clean}")

        return json.dumps({"validated": True, "step_number": step_number, "database": databaseName})
    except Exception as e:
        return json.dumps({"validated": False, "step_number": step_number,
                           "database": databaseName, "error": str(e)})










def _is_large_table(row_count: int, threshold: int = 10000) -> bool:
    return row_count >= threshold














@mcp.tool(name="get_table_stats", description="Get lightweight stats for one table.")
async def get_table_stats(ctx: Context[ServerSession, AppContext],
                          databaseName: str, schemaName: str, table_name: str) -> str:
    try:
        conn_id = ctx.request_context.lifespan_context.conn_ids[databaseName]
        rows = await ctx.request_context.lifespan_context.db.query(
            conn_id,
            """
            SELECT
                COALESCE(NULLIF(c.reltuples, -1), 0)::bigint AS estimated_row_count
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = $1
              AND c.relname = $2
              AND c.relkind IN ('r', 'p', 'm', 'f', 'v')
            LIMIT 1
            """,
            (schemaName, table_name),
        )
        rc = int(rows[0]["estimated_row_count"]) if rows else 0
        return json.dumps({
            "databaseName": databaseName,
            "schemaName": schemaName,
            "table_name": table_name,
            "row_count": rc,
            "row_count_is_estimate": True,
            "is_large_table": _is_large_table(rc),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _enforce_query_rules(sql: str, row_count: int) -> str | None:
    sql_upper = (sql or "").upper()
    if "SELECT *" in sql_upper:
        return "SELECT * is not allowed."
    is_aggregate_output = (
        "GROUP BY" in sql_upper
        or any(func in sql_upper for func in ("COUNT(", "SUM(", "AVG(", "MIN(", "MAX("))
    )
    if is_aggregate_output:
        return None
    if row_count >= 10000 and "LIMIT" not in sql_upper and "WHERE" not in sql_upper:
        return "Large unfiltered queries must include LIMIT or WHERE."
    return None
