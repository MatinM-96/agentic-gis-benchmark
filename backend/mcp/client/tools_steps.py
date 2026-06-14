from typing import Any

from langchain_core.tools import StructuredTool

from backend.mcp.client.client import mcp_call, mcp_client
from backend.mcp.client.parsing import _parse, _parse_step




# --------------------------------------------------
# Step tools — all use _parse_step (raises on executed=False)
# --------------------------------------------------

async def execute_query_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_query_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_query_step")


async def execute_filter_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_filter_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_filter_step")


async def execute_select_columns_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_select_columns_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_select_columns_step")


async def execute_spatial_join_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_spatial_join_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_spatial_join_step")


async def execute_attribute_join_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_attribute_join_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_attribute_join_step")


async def execute_buffer_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_buffer_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_buffer_step")


async def execute_nearest_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_nearest_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_nearest_step")


async def execute_distance_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_distance_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_distance_step")


async def execute_aggregate_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_aggregate_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_aggregate_step")


async def execute_sort_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_sort_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_sort_step")


async def execute_limit_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_limit_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_limit_step")


async def execute_clip_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_clip_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_clip_step")


async def execute_union_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_union_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_union_step")


async def execute_intersection_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_intersection_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_intersection_step")


async def execute_difference_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_difference_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_difference_step")


async def execute_dissolve_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_dissolve_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_dissolve_step")


async def execute_centroid_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_centroid_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_centroid_step")


async def execute_transform_crs_step(run_id: str, step: dict) -> dict:
    raw = await mcp_call("execute_transform_crs_step", {"run_id": run_id, "step": step})
    return _parse_step(raw, "execute_transform_crs_step")


async def get_step_result(run_id: str, step_no: int, sample_only: bool = False) -> dict:
    raw = await mcp_call("get_step_result", {"run_id": run_id, "step_no": step_no, "sample_only": sample_only})
    return _parse(raw, "get_step_result")





async def _cleanup_run(run_id: str) -> str:
    return await mcp_client.call("cleanup_run", {"run_id": run_id})


tool_cleanup_run = StructuredTool.from_function(
    coroutine=_cleanup_run,
    name="tool_cleanup_run",
    description="Delete temporary stored step results for a run_id on the MCP server.",
)

