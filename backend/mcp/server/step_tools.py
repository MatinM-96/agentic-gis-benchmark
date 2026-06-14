


import json
import logging
import os

import duckdb
import geopandas as gpd
import pandas as pd
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from backend.mcp.server.registry import mcp
from backend.mcp.server.discovery_tools import _enforce_query_rules
from backend.mcp.server.geometry_utils import _detect_geom_col, _sjoin_nearest_return_right, _to_df, _to_gdf
from backend.mcp.server.models import AppContext, PlanStep
from backend.mcp.server.step_storage import (
    _err,
    _get_rows,
    _get_step_entry,
    _ok,
    _sample_rows,
    _save_rows,
)

logger = logging.getLogger(__name__)

MAX_SPATIAL_TOOL_PAIR_CANDIDATES = int(
    os.getenv("MAX_SPATIAL_TOOL_PAIR_CANDIDATES", "500000000")
)


def _normalize_step(step: dict, infer_action: str | None = None) -> dict:
    """Return the step payload unchanged so MCP validates one strict schema only."""
    return dict(step)


def _stored_row_count(entry: dict) -> int:
    row_count = entry.get("row_count")
    if row_count is not None:
        return int(row_count)
    rows = entry.get("rows")
    return len(rows) if isinstance(rows, list) else 0


def _spatial_pair_guard(
    step_no: int,
    action: str,
    left_entry: dict,
    right_entry: dict,
    left_source: int,
    right_source: int,
) -> str | None:
    left_count = _stored_row_count(left_entry)
    right_count = _stored_row_count(right_entry)
    pair_count = left_count * right_count

    if pair_count <= MAX_SPATIAL_TOOL_PAIR_CANDIDATES:
        return None

    return _err(
        step_no,
        action,
        (
            f"{action} input too large before GeoPandas execution: "
            f"left step {left_source} has {left_count} rows and right step {right_source} "
            f"has {right_count} rows ({pair_count} possible pairs). "
            f"Limit is {MAX_SPATIAL_TOOL_PAIR_CANDIDATES} possible pairs. "
            "To fix: re-query the larger step with a spatial bounding-box filter — "
            "compute the bbox of the smaller result set and add "
            "ST_Intersects(geom, ST_MakeEnvelope(minx, miny, maxx, maxy, srid)) "
            "to that step's WHERE clause, then retry this spatial tool."
        ),
    )




@mcp.tool(
    name="cleanup_run",
    description="Delete temporary stored step results for a run_id, including DuckDB-backed intermediate tables.",
)
async def cleanup_run(ctx: Context[ServerSession, AppContext], run_id: str) -> str:
    lifespan = ctx.request_context.lifespan_context
    run_store = lifespan.step_results.get(run_id)

    if not run_store:
        return json.dumps({"ok": True, "run_id": run_id, "deleted_steps": 0,
                           "message": "No stored step data found for this run_id."})

    con = None
    deleted_steps = 0
    dropped_tables = []

    try:
        duckdb_path = getattr(lifespan, "duckdb_path", None)
        if duckdb_path:
            con = duckdb.connect(duckdb_path)

        for step_no, entry in run_store.items():
            deleted_steps += 1
            if not isinstance(entry, dict):
                continue
            if entry.get("storage") == "duckdb":
                table_name = entry.get("table")
                if con is not None and table_name:
                    con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                    dropped_tables.append(table_name)

        lifespan.step_results.pop(run_id, None)
        return json.dumps({"ok": True, "run_id": run_id, "deleted_steps": deleted_steps,
                           "dropped_tables": dropped_tables})
    except Exception as e:
        return json.dumps({"ok": False, "run_id": run_id, "error": str(e)})
    finally:
        if con is not None:
            con.close()


@mcp.tool(
    name="get_step_result",
    description="Fetch the full stored result for a previously executed step.",
)
async def get_step_result(ctx: Context[ServerSession, AppContext], run_id: str, step_no: int,
                          sample_only: bool = True) -> str:
    try:
        entry = _get_step_entry(ctx, run_id, step_no)
        rows = _get_rows(ctx, run_id, step_no)
        total = len(rows)
        columns = list(rows[0].keys()) if rows else []
        if sample_only:
            rows = _sample_rows(rows)
        return json.dumps({
            "run_id": run_id,
            "step": step_no,
            "row_count": total,
            "columns": columns,
            "rows": rows,
            "geom_col": entry.get("geom_col"),
            "crs": entry.get("crs"),
            "storage": entry.get("storage"),
        }, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

# --------------------------------------------------
# Step tools
# --------------------------------------------------

@mcp.tool(name="execute_query_step", description="Execute one validated SQL query step.")
async def execute_query_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step, infer_action="query"))
        if not s.database or not s.sql:
            return _err(s.step, "query", "step requires database and sql.")

        conn_id = ctx.request_context.lifespan_context.conn_ids[s.database]
        row_count = 0
        if s.schema_name and s.table_name:
            stats_rows = await ctx.request_context.lifespan_context.db.query(
                conn_id, f'SELECT COUNT(*) AS row_count FROM "{s.schema_name}"."{s.table_name}"'
            )
            row_count = int(stats_rows[0]["row_count"])

        rule_error = _enforce_query_rules(s.sql, row_count)
        if rule_error:
            return _err(s.step, "query", rule_error)

        rows = await ctx.request_context.lifespan_context.db.query(conn_id, s.sql)
        geom_col = _detect_geom_col(pd.DataFrame(rows)) if rows else None

        # Detect actual SRID from geometry_columns instead of hardcoding
        detected_crs: str | None = None
        if geom_col and s.schema_name and s.table_name:
            try:
                srid_rows = await ctx.request_context.lifespan_context.db.query(
                    conn_id,
                    f"""
                    SELECT srid FROM geometry_columns
                    WHERE f_table_schema = $1 AND f_table_name = $2
                      AND f_geometry_column = $3
                    LIMIT 1
                    """,
                    (s.schema_name, s.table_name, geom_col),
                )
                if srid_rows and srid_rows[0]["srid"] not in (0, None):
                    detected_crs = f"EPSG:{srid_rows[0]['srid']}"
            except Exception:
                pass  # Non-fatal — CRS will be None; downstream tools handle it

        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=geom_col,
                   crs=detected_crs,
                   step_meta={"action": "query", "sql": s.sql, "database": s.database,
                               "schema": s.schema_name, "table": s.table_name})
        return _ok(run_id, s.step, "query", rows)
    except Exception as e:
        logger.error("execute_query_step failed step=%s: %r", step_no, e)
        return _err(step_no, "query", repr(e))


@mcp.tool(
    name="execute_filter_step",
    description=(
        "Filter rows from a previous step using pandas query expressions. "
        "Supports standard comparisons and boolean logic only, e.g. "
        "'column > 100', 'status == \"active\"', 'area > 500 and risk == \"high\"'. "
        "Does NOT support PostGIS/SQL functions such as ST_IsValid, ST_Within etc. "
        "For spatial filtering, use execute_spatial_join_step instead."
    ),
)
async def execute_filter_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        df = _to_df(_get_rows(ctx, run_id, s.source_step))
        all_filters = s.filters
        if not df.empty and all_filters:
            for expr in all_filters:
                try:
                    df = df.query(expr, engine="python")
                except Exception as e:
                    return _err(
                        s.step,
                        "filter",
                        f"Invalid filter expression {expr!r}: {e}. Available columns: {list(df.columns)}",
                    )
        rows = df.to_dict("records")
        source_entry = _get_step_entry(ctx, run_id, s.source_step)
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=source_entry.get("geom_col"),
                   crs=source_entry.get("crs"),
                   step_meta={"action": "filter", "source_step": s.source_step, "filters": all_filters})
        return _ok(run_id, s.step, "filter", rows)
    except Exception as e:
        return _err(step_no, "filter", str(e))


@mcp.tool(name="execute_select_columns_step", description="Select specific columns from a previous step.")
async def execute_select_columns_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        df = _to_df(_get_rows(ctx, run_id, s.source_step))
        cols = s.select_columns
        if not df.empty and cols:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                return _err(s.step, "select_columns",
                            f"Columns not found: {missing}. Available: {list(df.columns)}")
            df = df[cols]
        rows = df.to_dict("records")
        source_entry = _get_step_entry(ctx, run_id, s.source_step)
        src_geom = source_entry.get("geom_col")
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=src_geom if (src_geom and (not cols or src_geom in cols)) else None,
                   crs=source_entry.get("crs") if src_geom and (not cols or src_geom in cols) else None,
                   step_meta={"action": "select_columns", "source_step": s.source_step, "columns": cols})
        return _ok(run_id, s.step, "select_columns", rows)
    except Exception as e:
        return _err(step_no, "select_columns", str(e))

















@mcp.tool(name="execute_spatial_join_step", description="Spatial join between two previous step results.")
async def execute_spatial_join_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step, infer_action="spatial_join"))
        if s.left_source is None or s.right_source is None:
            return _err(step_no, "spatial_join", "left_source and right_source are required integers.")
        if s.join_type not in {"intersects", "within", "contains", "dwithin", "touches", "overlaps"}:
            return _err(step_no, "spatial_join", f"Invalid join_type: {s.join_type}")

        left_entry = _get_step_entry(ctx, run_id, s.left_source)
        right_entry = _get_step_entry(ctx, run_id, s.right_source)
        guard_error = _spatial_pair_guard(
            step_no, "spatial_join", left_entry, right_entry, s.left_source, s.right_source
        )
        if guard_error:
            return guard_error

        left_gdf = _to_gdf(_get_rows(ctx, run_id, s.left_source),
                           s.left_geom or left_entry.get("geom_col"), left_entry.get("crs"))
        right_gdf = _to_gdf(_get_rows(ctx, run_id, s.right_source),
                            s.right_geom or right_entry.get("geom_col"), right_entry.get("crs"))

        if left_gdf.crs and right_gdf.crs and left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)

        if left_gdf.empty or right_gdf.empty:
            rows = []
        elif s.join_type == "dwithin":
            if s.distance_meters is None:
                return _err(step_no, "spatial_join", "distance_meters is required for dwithin.")
            joined = gpd.sjoin(left_gdf, right_gdf, predicate="dwithin", how="inner",
                               distance=s.distance_meters)
            rows = joined.drop(columns=["index_right"], errors="ignore").to_dict("records")
        else:
            joined = gpd.sjoin(left_gdf, right_gdf, predicate=s.join_type, how="inner")
            rows = joined.drop(columns=["index_right"], errors="ignore").to_dict("records")

        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=left_gdf.geometry.name,
                   crs=str(left_gdf.crs) if left_gdf.crs else None,
                   step_meta={"action": "spatial_join", "join_type": s.join_type,
                               "left_source": s.left_source, "right_source": s.right_source,
                               "distance_meters": s.distance_meters})
        return _ok(run_id, s.step, "spatial_join", rows)
    except Exception as e:
        return _err(step_no, "spatial_join", str(e))









@mcp.tool(name="execute_attribute_join_step", description="Attribute join between two previous step results.")
async def execute_attribute_join_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        keys = s.join_keys
        if not keys:
            return _err(step_no, "attribute_join", "join_keys is required.")
        left_df = _to_df(_get_rows(ctx, run_id, s.left_source))
        right_df = _to_df(_get_rows(ctx, run_id, s.right_source))
        rows = [] if (left_df.empty or right_df.empty) else \
            left_df.merge(right_df, on=keys, how="inner").to_dict("records")
        left_entry = _get_step_entry(ctx, run_id, s.left_source)
        src_geom = left_entry.get("geom_col")
        result_cols = list(rows[0].keys()) if rows else []
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=src_geom if src_geom and src_geom in result_cols else None,
                   crs=left_entry.get("crs") if src_geom and src_geom in result_cols else None,
                   step_meta={"action": "attribute_join", "left_source": s.left_source,
                               "right_source": s.right_source, "join_keys": keys})
        return _ok(run_id, s.step, "attribute_join", rows)
    except Exception as e:
        return _err(step_no, "attribute_join", str(e))












@mcp.tool(name="execute_buffer_step", description="Buffer geometries from a previous step.")
async def execute_buffer_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        entry = _get_step_entry(ctx, run_id, s.source_step)
        gdf = _to_gdf(_get_rows(ctx, run_id, s.source_step),
                      s.left_geom or entry.get("geom_col"), entry.get("crs")).copy()
        geom_col = gdf.geometry.name
        if not gdf.empty:
            gdf[geom_col] = gdf.geometry.buffer(s.buffer_distance)
            gdf = gdf.set_geometry(geom_col)
        rows = gdf.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=geom_col, crs=str(gdf.crs) if gdf.crs else None,
                   step_meta={"action": "buffer", "source_step": s.source_step,
                               "buffer_distance": s.buffer_distance})
        return _ok(run_id, s.step, "buffer", rows)
    except Exception as e:
        return _err(step_no, "buffer", str(e))


















@mcp.tool(name="execute_nearest_step", description="Find nearest geometry from another step.")
async def execute_nearest_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        left_entry = _get_step_entry(ctx, run_id, s.left_source)
        right_entry = _get_step_entry(ctx, run_id, s.right_source)
        guard_error = _spatial_pair_guard(
            step_no, "nearest", left_entry, right_entry, s.left_source, s.right_source
        )
        if guard_error:
            return guard_error

        left_gdf = _to_gdf(_get_rows(ctx, run_id, s.left_source),
                           s.left_geom or left_entry.get("geom_col"), left_entry.get("crs"))
        right_gdf = _to_gdf(_get_rows(ctx, run_id, s.right_source),
                            s.right_geom or right_entry.get("geom_col"), right_entry.get("crs"))
        if left_gdf.crs and right_gdf.crs and left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)
        rows = _sjoin_nearest_return_right(
            left_gdf, right_gdf, distance_col=s.metric_as or "distance"
        )
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=right_gdf.geometry.name, crs=str(right_gdf.crs) if right_gdf.crs else None,
                   step_meta={"action": "nearest", "left_source": s.left_source,
                               "right_source": s.right_source})
        return _ok(run_id, s.step, "nearest", rows)
    except Exception as e:
        return _err(step_no, "nearest", str(e))

















@mcp.tool(name="execute_distance_step", description="Compute nearest distance to another step.")
async def execute_distance_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        left_entry = _get_step_entry(ctx, run_id, s.left_source)
        right_entry = _get_step_entry(ctx, run_id, s.right_source)
        guard_error = _spatial_pair_guard(
            step_no, "distance", left_entry, right_entry, s.left_source, s.right_source
        )
        if guard_error:
            return guard_error

        left_gdf = _to_gdf(_get_rows(ctx, run_id, s.left_source),
                           s.left_geom or left_entry.get("geom_col"), left_entry.get("crs"))
        right_gdf = _to_gdf(_get_rows(ctx, run_id, s.right_source),
                            s.right_geom or right_entry.get("geom_col"), right_entry.get("crs"))
        if left_gdf.crs and right_gdf.crs and left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)
        rows = [] if (left_gdf.empty or right_gdf.empty) else \
            gpd.sjoin_nearest(left_gdf, right_gdf, how="left", distance_col="distance_m").to_dict("records")
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=left_gdf.geometry.name, crs=str(left_gdf.crs) if left_gdf.crs else None,
                   step_meta={"action": "distance", "left_source": s.left_source,
                               "right_source": s.right_source})
        return _ok(run_id, s.step, "distance", rows)
    except Exception as e:
        return _err(step_no, "distance", str(e))









@mcp.tool(name="execute_aggregate_step", description="Aggregate rows from a previous step.")
async def execute_aggregate_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        df = _to_df(_get_rows(ctx, run_id, s.source_step))
        agg_type = s.aggregation_type or "count"
        metric_col = s.metric_column
        metric_as = s.metric_as or "metric"
        order_by = s.order_by
        order_direction = s.order_direction or "asc"

        if agg_type != "count" and not metric_col:
            return _err(s.step, "aggregate", f"aggregation_type '{agg_type}' requires metric_column.")

        if df.empty:
            out = pd.DataFrame()
        elif agg_type == "count":
            out = (df.groupby(s.group_by).size().reset_index(name=metric_as)
                   if s.group_by else pd.DataFrame([{metric_as: len(df)}]))
        else:
            out = (df.groupby(s.group_by)[metric_col].agg(agg_type).reset_index(name=metric_as)
                   if s.group_by
                   else pd.DataFrame([{metric_as: getattr(df[metric_col], agg_type)()}]))

        if not out.empty and order_by and order_by in out.columns:
            out = out.sort_values(order_by, ascending=(order_direction == "asc"))
        if not out.empty and s.limit is not None:
            out = out.head(s.limit)

        rows = out.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows,
                   step_meta={"action": "aggregate", "aggregation_type": agg_type,
                               "group_by": s.group_by, "metric_column": s.metric_column,
                               "metric_as": metric_as, "source_step": s.source_step})
        return _ok(run_id, s.step, "aggregate", rows)
    except Exception as e:
        return _err(step_no, "aggregate", str(e))
















@mcp.tool(name="execute_sort_step", description="Sort rows from a previous step.")
async def execute_sort_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        df = _to_df(_get_rows(ctx, run_id, s.source_step))
        if df.empty:
            _save_rows(ctx, run_id, s.step, [])
            return _ok(run_id, s.step, "sort", [])

        order_by = s.order_by
        order_direction = s.order_direction or "asc"

        if not order_by or order_by not in df.columns:
            return _err(s.step, "sort", f"Column '{order_by}' not found. Available: {list(df.columns)}")
        df = df.sort_values(order_by, ascending=(order_direction == "asc"))
        rows = df.to_dict("records")
        source_entry = _get_step_entry(ctx, run_id, s.source_step)
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=source_entry.get("geom_col"),
                   crs=source_entry.get("crs"),
                   step_meta={"action": "sort", "source_step": s.source_step,
                               "order_by": order_by, "order_direction": order_direction})
        return _ok(run_id, s.step, "sort", rows)
    except Exception as e:
        return _err(step_no, "sort", str(e))















@mcp.tool(name="execute_limit_step", description="Limit rows from a previous step.")
async def execute_limit_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        if s.limit is None:
            return _err(s.step, "limit", "limit step requires limit.")
        df = _to_df(_get_rows(ctx, run_id, s.source_step))
        rows = df.head(s.limit).to_dict("records")
        source_entry = _get_step_entry(ctx, run_id, s.source_step)
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=source_entry.get("geom_col"),
                   crs=source_entry.get("crs"),
                   step_meta={"action": "limit", "source_step": s.source_step, "limit": s.limit})
        return _ok(run_id, s.step, "limit", rows)
    except Exception as e:
        return _err(step_no, "limit", str(e))
























@mcp.tool(name="execute_clip_step", description="Clip one geometry layer by another.")
async def execute_clip_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        left_entry = _get_step_entry(ctx, run_id, s.left_source)
        right_entry = _get_step_entry(ctx, run_id, s.right_source)
        guard_error = _spatial_pair_guard(
            step_no, "clip", left_entry, right_entry, s.left_source, s.right_source
        )
        if guard_error:
            return guard_error
        left_gdf = _to_gdf(_get_rows(ctx, run_id, s.left_source),
                           s.left_geom or left_entry.get("geom_col"), left_entry.get("crs"))
        right_gdf = _to_gdf(_get_rows(ctx, run_id, s.right_source),
                            s.right_geom or right_entry.get("geom_col"), right_entry.get("crs"))
        if left_gdf.crs and right_gdf.crs and left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)
        rows = [] if (left_gdf.empty or right_gdf.empty) else \
            gpd.clip(left_gdf, right_gdf).to_dict("records")
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=left_gdf.geometry.name, crs=str(left_gdf.crs) if left_gdf.crs else None,
                   step_meta={"action": "clip", "left_source": s.left_source,
                               "right_source": s.right_source})
        return _ok(run_id, s.step, "clip", rows)
    except Exception as e:
        return _err(step_no, "clip", str(e))


















@mcp.tool(name="execute_union_step", description="Union all geometries from a previous step.")
async def execute_union_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        entry = _get_step_entry(ctx, run_id, s.source_step)
        gdf = _to_gdf(_get_rows(ctx, run_id, s.source_step),
                      s.left_geom or entry.get("geom_col"), entry.get("crs"))
        geom_col = gdf.geometry.name
        rows = [] if gdf.empty else [{geom_col: gdf.geometry.union_all().wkb.hex()}]
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=geom_col, crs=str(gdf.crs) if gdf.crs else None,
                   step_meta={"action": "union", "source_step": s.source_step})
        return _ok(run_id, s.step, "union", rows)
    except Exception as e:
        return _err(step_no, "union", str(e))

















@mcp.tool(name="execute_intersection_step", description="Overlay intersection between two geometry layers.")
async def execute_intersection_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        left_entry = _get_step_entry(ctx, run_id, s.left_source)
        right_entry = _get_step_entry(ctx, run_id, s.right_source)
        guard_error = _spatial_pair_guard(
            step_no, "intersection", left_entry, right_entry, s.left_source, s.right_source
        )
        if guard_error:
            return guard_error
        left_gdf = _to_gdf(_get_rows(ctx, run_id, s.left_source),
                           s.left_geom or left_entry.get("geom_col"), left_entry.get("crs"))
        right_gdf = _to_gdf(_get_rows(ctx, run_id, s.right_source),
                            s.right_geom or right_entry.get("geom_col"), right_entry.get("crs"))
        if left_gdf.crs and right_gdf.crs and left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)
        rows = [] if (left_gdf.empty or right_gdf.empty) else \
            gpd.overlay(left_gdf, right_gdf, how="intersection").to_dict("records")
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=left_gdf.geometry.name, crs=str(left_gdf.crs) if left_gdf.crs else None,
                   step_meta={"action": "intersection", "left_source": s.left_source,
                               "right_source": s.right_source})
        return _ok(run_id, s.step, "intersection", rows)
    except Exception as e:
        return _err(step_no, "intersection", str(e))

















@mcp.tool(name="execute_difference_step", description="Overlay difference between two geometry layers.")
async def execute_difference_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        left_entry = _get_step_entry(ctx, run_id, s.left_source)
        right_entry = _get_step_entry(ctx, run_id, s.right_source)
        guard_error = _spatial_pair_guard(
            step_no, "difference", left_entry, right_entry, s.left_source, s.right_source
        )
        if guard_error:
            return guard_error
        left_gdf = _to_gdf(_get_rows(ctx, run_id, s.left_source),
                           s.left_geom or left_entry.get("geom_col"), left_entry.get("crs"))
        right_gdf = _to_gdf(_get_rows(ctx, run_id, s.right_source),
                            s.right_geom or right_entry.get("geom_col"), right_entry.get("crs"))
        if left_gdf.crs and right_gdf.crs and left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)
        rows = [] if (left_gdf.empty or right_gdf.empty) else \
            gpd.overlay(left_gdf, right_gdf, how="difference").to_dict("records")
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=left_gdf.geometry.name, crs=str(left_gdf.crs) if left_gdf.crs else None,
                   step_meta={"action": "difference", "left_source": s.left_source,
                               "right_source": s.right_source})
        return _ok(run_id, s.step, "difference", rows)
    except Exception as e:
        return _err(step_no, "difference", str(e))

















@mcp.tool(name="execute_dissolve_step", description="Dissolve geometries by group.")
async def execute_dissolve_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        entry = _get_step_entry(ctx, run_id, s.source_step)
        gdf = _to_gdf(_get_rows(ctx, run_id, s.source_step),
                      s.left_geom or entry.get("geom_col"), entry.get("crs"))
        if gdf.empty:
            rows = []
        else:
            out = gdf.dissolve(by=s.group_by[0]).reset_index() if s.group_by else gdf.dissolve().reset_index()
            rows = out.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=gdf.geometry.name, crs=str(gdf.crs) if gdf.crs else None,
                   step_meta={"action": "dissolve", "source_step": s.source_step,
                               "group_by": s.group_by})
        return _ok(run_id, s.step, "dissolve", rows)
    except Exception as e:
        return _err(step_no, "dissolve", str(e))












@mcp.tool(name="execute_centroid_step", description="Compute centroids for a geometry layer.")
async def execute_centroid_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        entry = _get_step_entry(ctx, run_id, s.source_step)
        gdf = _to_gdf(_get_rows(ctx, run_id, s.source_step),
                      s.left_geom or entry.get("geom_col"), entry.get("crs")).copy()
        geom_col = gdf.geometry.name
        if not gdf.empty:
            gdf[geom_col] = gdf.geometry.centroid
            gdf = gdf.set_geometry(geom_col)
        rows = gdf.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=geom_col, crs=str(gdf.crs) if gdf.crs else None,
                   step_meta={"action": "centroid", "source_step": s.source_step})
        return _ok(run_id, s.step, "centroid", rows)
    except Exception as e:
        return _err(step_no, "centroid", str(e))


@mcp.tool(name="execute_transform_crs_step", description="Transform geometries to another CRS.")
async def execute_transform_crs_step(ctx: Context[ServerSession, AppContext], run_id: str, step: dict) -> str:
    step_no = step.get("step") if isinstance(step, dict) else -1
    try:
        s = PlanStep.model_validate(_normalize_step(step))
        entry = _get_step_entry(ctx, run_id, s.source_step)
        gdf = _to_gdf(_get_rows(ctx, run_id, s.source_step),
                      s.left_geom or entry.get("geom_col"), entry.get("crs"))
        if not gdf.empty:
            gdf = gdf.to_crs(s.target_crs)
        rows = gdf.to_dict("records")
        _save_rows(ctx, run_id, s.step, rows,
                   geom_col=gdf.geometry.name, crs=str(gdf.crs) if gdf.crs else None,
                   step_meta={"action": "transform_crs", "source_step": s.source_step,
                               "target_crs": s.target_crs})
        return _ok(run_id, s.step, "transform_crs", rows)
    except Exception as e:
        return _err(step_no, "transform_crs", str(e))
