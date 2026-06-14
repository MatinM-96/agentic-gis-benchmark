import json
from typing import Any


def _parse(raw: str, tool_name: str) -> Any:
    if not raw:
        raise RuntimeError(f"Empty response from {tool_name}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{tool_name} returned non-JSON: {raw[:200]}") from e


def _parse_step(raw: str, tool_name: str) -> dict:
    result = _parse(raw, tool_name)
    if not isinstance(result, dict):
        raise RuntimeError(f"{tool_name} returned unexpected type: {type(result)}")
    if not result.get("executed", False):
        raise RuntimeError(f"{tool_name} failed: {result.get('error', 'unknown error')}")
    return result

