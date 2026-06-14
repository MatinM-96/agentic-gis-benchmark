from backend.mcp.client.client import mcp_call
from backend.mcp.client.parsing import _parse




async def get_available_databases() -> list[str]:
    raw = await mcp_call("get_available_databases", {})
    data = _parse(raw, "get_available_databases")
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(data["error"])
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response from get_available_databases: {data}")
    return data




async def prefetch_databases(database_names: list[str]) -> dict:
    raw = await mcp_call("prefetch_databases", {"databaseNames": database_names})
    return _parse(raw, "prefetch_databases")


async def get_available_schemas(databaseName: str) -> dict:
    raw = await mcp_call("get_available_schemas", {"databaseName": databaseName})
    return _parse(raw, "get_available_schemas")


async def get_available_tables(databaseName: str, schemaName: str) -> dict:
    raw = await mcp_call("get_available_tables", {"databaseName": databaseName, "schemaName": schemaName})
    return _parse(raw, "get_available_tables")


async def get_table_schema(databaseName: str, schemaName: str, table_name: str) -> dict:
    raw = await mcp_call("get_table_schema", {"databaseName": databaseName, "schemaName": schemaName, "table_name": table_name})
    return _parse(raw, "get_table_schema")


async def get_database_catalog(databaseName: str | None = None, schemaName: str | None = None) -> dict:
    args = {}
    if databaseName is not None:
        args["databaseName"] = databaseName
    if schemaName is not None:
        args["schemaName"] = schemaName
    raw = await mcp_call("get_database_catalog", args)
    return _parse(raw, "get_database_catalog")

async def validate_sql_via_mcp(sql: str, databaseName: str, step_number: int) -> dict:
    raw = await mcp_call(
        "validate_sql_explain",
        {"sql": sql, "databaseName": databaseName, "step_number": step_number},
    )
    result = _parse(raw, "validate_sql_explain")
    if not isinstance(result, dict):
        raise RuntimeError(f"validate_sql_explain returned unexpected type: {type(result)}")
    return result
