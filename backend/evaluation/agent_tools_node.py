import hashlib
import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

from agent_models import AgentState
from backend.evaluation.prompts.agent_helpers import (
    _coerce_args,
    _dedup,
    _extract_databases_from_result,
    _extract_from_database_catalog,
    _extract_tables_from_available_tables_result,
    _increment_sql_repair,
    _increment_step_retry,
    _parse_tool_result,
    _validated_sql_key,
    _build_snapshot,
    _sql_repair_ok,
    _step_retry_ok,
)
from backend.tools.langchain_tools import (
    tool_get_available_databases,
    tool_get_available_tables,
    tool_get_table_schema,
    tool_get_database_catalog,
    tool_get_route,
    tool_validate_sql,


    
    tool_execute_query_step,
    tool_execute_filter_step,
    tool_execute_select_columns_step,
    tool_execute_spatial_join_step,
    tool_execute_attribute_join_step,
    tool_execute_buffer_step,
    tool_execute_nearest_step,
    tool_execute_distance_step,
    tool_execute_aggregate_step,
    tool_execute_sort_step,
    tool_execute_limit_step,
    tool_execute_clip_step,
    tool_execute_union_step,
    tool_execute_intersection_step,
    tool_execute_difference_step,
    tool_execute_dissolve_step,
    tool_execute_centroid_step,
    tool_execute_transform_crs_step,
)












STEP_TOOLS: frozenset[str] = frozenset({
    "tool_execute_query_step",
    "tool_execute_filter_step",
    "tool_execute_select_columns_step",
    "tool_execute_spatial_join_step",
    "tool_execute_attribute_join_step",
    "tool_execute_buffer_step",
    "tool_execute_nearest_step",
    "tool_execute_distance_step",
    "tool_execute_aggregate_step",
    "tool_execute_sort_step",
    "tool_execute_limit_step",
    "tool_execute_clip_step",
    "tool_execute_union_step",
    "tool_execute_intersection_step",
    "tool_execute_difference_step",
    "tool_execute_dissolve_step",
    "tool_execute_centroid_step",
    "tool_execute_transform_crs_step",
})

NON_DISCOVERY_EXEC_TOOLS: frozenset[str] = frozenset()




STEP_DICT_FIELDS: frozenset[str] = frozenset({
    "step", "action", "database", "sql", "validSql",
    "source_step", "left_source", "right_source",
    "join_type", "left_geom", "right_geom",
    "distance_meters", "buffer_distance",
    "filters", "select_columns", "output_columns",
    "group_by", "aggregation_type", "metric_column", "metric_as",
    "order_by", "order_direction", "limit",
    "target_crs", "join_keys",
    "schema_name", "table_name",
})





# Actions that require source_step
SOURCE_STEP_ACTIONS: frozenset[str] = frozenset({
    "filter", "select_columns", "sort", "limit",
    "buffer", "centroid", "transform_crs", "dissolve", "union", "aggregate",
})

# Actions that require left_source + right_source
DUAL_SOURCE_ACTIONS: frozenset[str] = frozenset({
    "spatial_join", "intersection", "difference",
    "clip", "nearest", "distance", "attribute_join",
})









def _zero_rows_sig(step_obj: dict) -> str:
    payload = {
        "action": step_obj.get("action"),
        "database": step_obj.get("database"),
        "sql": " ".join((step_obj.get("sql") or "").lower().split()),
        "source_step": step_obj.get("source_step"),
        "left_source": step_obj.get("left_source"),
        "right_source": step_obj.get("right_source"),
        "join_type": step_obj.get("join_type"),
        "order_by": step_obj.get("order_by"),
        "order_direction": step_obj.get("order_direction"),
        "limit": step_obj.get("limit"),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:16]

def _normalize_tool_args(name: str, raw_args: dict, run_id: str) -> dict:
    """Inject run_id into step tool args. Model must send correct shape."""
    args = dict(raw_args or {})
    if name in STEP_TOOLS:
        args["run_id"] = run_id
    return args


def _block_step_result(name: str, reason: str, step_number: int | None, action: str | None) -> str:
    return json.dumps({
        "executed": False,
        "error": f"{name} blocked.",
        "reason": reason,
        "step": step_number,
        "action": action,
    })


def _lock_step(state: "AgentState", step_number: int | None) -> None:
    if step_number is None:
        return

    locked_step_ids = set(state.get("locked_step_ids", set()))
    locked_step_ids.add(step_number)
    state["locked_step_ids"] = locked_step_ids

    step_registry = dict(state.get("step_registry", {}))
    prev = dict(step_registry.get(step_number, {}))
    if prev:
        prev["locked"] = True
        step_registry[step_number] = prev
        state["step_registry"] = step_registry



def _planned_step_meta(state: "AgentState", step_number: int | None) -> dict:
    if step_number is None:
        return {}

    planned_steps = state.get("planned_steps", {}) or {}
    planned = planned_steps.get(step_number)
    if planned is None:
        planned = planned_steps.get(str(step_number))
    return dict(planned) if isinstance(planned, dict) else {}


def _planned_query_database(planned_step: dict) -> str:
    return (planned_step.get("database") or "").strip() if isinstance(planned_step, dict) else ""


def _is_terminal_required_step(state: "AgentState", step_number: int | None) -> bool:
    if step_number is None:
        return False

    required = set(state.get("required_step_ids", set()))
    if not required:
        return False

    current = str(step_number)
    required_keys = {str(step_id) for step_id in required}
    if current not in required_keys:
        return False

    numeric_required: list[int] = []
    for step_id in required_keys:
        if step_id.isdigit():
            numeric_required.append(int(step_id))

    if numeric_required and current.isdigit():
        return int(current) == max(numeric_required)

    return current == sorted(required_keys)[-1]


def _is_intermediate_empty_step(
    state: "AgentState",
    step_number: int | None,
    rows_returned: int,
) -> bool:
    return rows_returned == 0 and not _is_terminal_required_step(state, step_number)


def _plan_mismatch_replan_payload(
    state: "AgentState",
    step_number: int | None,
    attempted_action: str,
    attempted_database: str = "",
) -> dict | None:
    if (
        not state.get("plan_finalized")
        or step_number is None
        or state.get("replan_trigger") is not None
    ):
        return None

    planned_step = _planned_step_meta(state, step_number)
    if not planned_step:
        return {
            "reason": "blocked_step",
            "error": (
                f"Step {step_number} is not part of the finalized plan. "
                "Produce a new numbered plan before continuing."
            ),
            "zero_rows_step": None,
        }

    planned_action = (planned_step.get("action") or "").strip()
    if planned_action and planned_action != attempted_action:
        return {
            "reason": "blocked_step",
            "error": (
                f"Step {step_number} is planned as '{planned_action}', "
                f"but was attempted as '{attempted_action}'. "
                "Produce a new plan or execute the planned action."
            ),
            "zero_rows_step": None,
        }

    planned_database = _planned_query_database(planned_step)
    if (
        planned_action == "query"
        and planned_database
        and attempted_database
        and planned_database != attempted_database
    ):
        return {
            "reason": "blocked_step",
            "error": (
                f"Step {step_number} is planned to query database '{planned_database}', "
                f"but was attempted against '{attempted_database}'. "
                "Produce a new plan or keep this step on the planned database."
            ),
            "zero_rows_step": None,
        }

    return None


def _guard_validate_sql(state: "AgentState", args: dict) -> str | None:
    sql = (args.get("sql") or "").strip()
    attempted_database = (args.get("databaseName") or "").strip()
    step_number = args.get("step_number")
    if step_number is None or not sql or not attempted_database:
        counts = dict(state.get("tool_shape_error_counts", {}))
        counts["tool_validate_sql"] = counts.get("tool_validate_sql", 0) + 1
        state["tool_shape_error_counts"] = counts
        missing = []
        if not sql:
            missing.append("sql")
        if not attempted_database:
            missing.append("databaseName")
        if step_number is None:
            missing.append("step_number")
        missing_str = ", ".join(missing)
        return _block_step_result(
            "tool_validate_sql",
            (
                f"tool_validate_sql requires {missing_str}. "
                "Correct shape: tool_validate_sql(sql='SELECT ...', "
                "databaseName='kommuner', step_number=1)"
            ),
            step_number,
            "query",
        )

    required_step_ids = {
        str(step_id) for step_id in set(state.get("required_step_ids", set()))
    }

    if state.get("replan_trigger") is not None:
        return _block_step_result(
            "tool_validate_sql",
            "Replan is required before further validation. Produce a new numbered plan first.",
            step_number,
            "query",
        )

    max_repairs = getattr(state.get("params"), "max_sql_repairs", 3)
    if not _sql_repair_ok(state, step_number, max_repairs):
        state["replan_trigger"] = {
            "reason": "blocked_step",
            "error": f"Step {step_number} exhausted SQL repair budget ({max_repairs} attempts).",
            "zero_rows_step": None,
        }
        return _block_step_result(
            "tool_validate_sql",
            f"Step {step_number} has exhausted its SQL repair budget ({max_repairs} attempts). Produce a new plan.",
            step_number,
            "query",
        )

    if not state.get("plan_finalized", False):
        count = state.get("plan_block_count", 0) + 1
        state["plan_block_count"] = count
        if count >= 3:
            state["replan_trigger"] = {
                "reason": "blocked_step",
                "error": "Plan not finalized after 3 validation blocks.",
                "zero_rows_step": None,
            }
            state["plan_block_count"] = 0
        return _block_step_result(
            "tool_validate_sql",
            "Planning is required before validation. First produce an explicit numbered plan for the task.",
            step_number,
            "query",
        )

    planned_steps = state.get("planned_steps", {})
    planned_keys = {str(k) for k in planned_steps.keys()}
    if planned_keys and str(step_number) not in planned_keys:
        return _block_step_result(
            "tool_validate_sql",
            f"Step {step_number} is not part of the finalized plan. Validate only planned step numbers.",
            step_number,
            "query",
        )

    planned_step = _planned_step_meta(state, step_number)
    planned_action = planned_step.get("action")
    if planned_action and planned_action != "query":
        return _block_step_result(
            "tool_validate_sql",
            f"Step {step_number} is planned as '{planned_action}', not a SQL query. Do not validate SQL for this step.",
            step_number,
            planned_action,
        )

    planned_database = _planned_query_database(planned_step)
    if planned_database and attempted_database and planned_database != attempted_database:
        return _block_step_result(
            "tool_validate_sql",
            (
                f"Step {step_number} is planned to query database '{planned_database}', "
                f"not '{attempted_database}'. Use the planned database for this step or "
                "produce a new plan with different step numbers."
            ),
            step_number,
            "query",
        )

    step_registry = dict(state.get("step_registry", {}))
    prev = dict(step_registry.get(step_number, {}))

    if step_number in set(state.get("locked_step_ids", set())):
        counts = dict(state.get("locked_step_block_counts", {}))
        counts[str(step_number)] = counts.get(str(step_number), 0) + 1
        state["locked_step_block_counts"] = counts
        return _block_step_result(
            "tool_validate_sql",
            f"Step {step_number} is locked after a successful validation/execution and cannot be changed.",
            step_number,
            prev.get("action", "query"),
        )

    prev_action = prev.get("action")
    if prev_action and prev_action != "query":
        return _block_step_result(
            "tool_validate_sql",
            f"Step {step_number} is already registered as action '{prev_action}' and cannot be changed to 'query'.",
            step_number,
            prev_action,
        )

    prev_database = (prev.get("database") or "").strip()
    if prev_action == "query" and prev_database and attempted_database and prev_database != attempted_database:
        return _block_step_result(
            "tool_validate_sql",
            (
                f"Step {step_number} is already registered for database '{prev_database}' "
                f"and cannot be changed to '{attempted_database}'."
            ),
            step_number,
            "query",
        )

    return None


def _guard_step_tool(
    state: "AgentState",
    name: str,
    raw_args: dict,
    args: dict,
    validated_sqls: dict,
) -> tuple[str | None, dict | None]:
    action_name = name.replace("tool_execute_", "").replace("_step", "")
    step_obj = args.get("step")
    run_id = args.get("run_id")

    def _flat_hint() -> str:
        flat_keys = STEP_DICT_FIELDS - {"step"}
        if any(k in raw_args for k in flat_keys):
            nested = {k: v for k, v in raw_args.items() if k != "run_id"}
            return (
                "You sent step fields as flat top-level arguments — "
                "they must be nested inside a single 'step' dict. "
                f"You sent: {json.dumps(raw_args, default=str)[:300]}. "
                f"Correct shape: {name}(run_id='<run_id>', "
                f"step={json.dumps(nested, default=str)[:200]})"
            )
        return (
            f"The 'step' argument must be a dict, not "
            f"{type(step_obj).__name__} = {repr(step_obj)[:100]}. "
            f"Correct shape: {name}(run_id='<run_id>', "
            f"step={{'step': <int>, 'action': '{action_name}', ...}})"
        )

    if not isinstance(step_obj, dict):
        counts = dict(state.get("tool_shape_error_counts", {}))
        counts[name] = counts.get(name, 0) + 1
        state["tool_shape_error_counts"] = counts
        return _block_step_result(name, _flat_hint(), None, action_name), None

    step_obj = dict(step_obj)
    step_number = step_obj.get("step")
    action = step_obj.get("action") or action_name
    step_obj["action"] = action
    sql = (step_obj.get("sql") or "").strip()
    database = (step_obj.get("database") or "").strip()
    blocked_reason: str | None = None
    prev = dict(state.get("step_registry", {}).get(step_number, {})) if step_number is not None else {}
    prev_action = prev.get("action")
    prev_database = (prev.get("database") or "").strip()

    required_step_ids = {
        str(step_id) for step_id in set(state.get("required_step_ids", set()))
    }
    planned_steps = state.get("planned_steps", {})
    planned_step = _planned_step_meta(state, step_number)
    planned_action = (planned_step.get("action") or "").strip()
    planned_database = _planned_query_database(planned_step)

    if name == "tool_execute_query_step":
        logger.info(
            "      [plan-guard] execute step=%s plan_finalized=%s required_step_ids=%s planned_steps=%s",
            step_number,
            state.get("plan_finalized", False),
            sorted(required_step_ids),
            sorted(str(k) for k in dict(state.get("planned_steps", {})).keys()),
        )

    if state.get("replan_trigger") is not None:
        blocked_reason = (
            "Replan is required before further execution. Produce a new numbered plan first."
        )
    elif not state.get("plan_finalized", False):
        count = state.get("plan_block_count", 0) + 1
        state["plan_block_count"] = count
        if count >= 3:
            state["replan_trigger"] = {
                "reason": "blocked_step",
                "error": "Plan not finalized after 3 execution blocks.",
                "zero_rows_step": None,
            }
            state["plan_block_count"] = 0
        blocked_reason = (
            "Planning is required before execution. First produce an explicit numbered plan for the task."
        )
    elif step_number is not None and planned_steps and str(step_number) not in {str(k) for k in planned_steps.keys()}:
        unplanned_counts = dict(state.get("unplanned_step_block_counts", {}))
        key = str(step_number)
        unplanned_counts[key] = unplanned_counts.get(key, 0) + 1
        state["unplanned_step_block_counts"] = unplanned_counts
        max_unplanned = getattr(state.get("params"), "max_step_retries", 3)
        if unplanned_counts[key] >= max_unplanned and state.get("replan_trigger") is None:
            state["replan_trigger"] = {
                "reason": "unplanned_step",
                "error": (
                    f"Step {step_number} was attempted {unplanned_counts[key]} times "
                    "but is not in the finalized plan."
                ),
                "zero_rows_step": None,
                "action": action_name,
                "blocked_count": unplanned_counts[key],
            }
            blocked_reason = (
                f"Step {step_number} is not in the finalized plan and has been blocked "
                f"{unplanned_counts[key]} times. You must produce a new plan that includes "
                "all required steps before executing."
            )
        else:
            blocked_reason = (
                f"Step {step_number} is not part of the finalized plan. "
                "Execute only planned step numbers."
            )
    elif step_number in set(state.get("locked_step_ids", set())):
        counts = dict(state.get("locked_step_block_counts", {}))
        counts[str(step_number)] = counts.get(str(step_number), 0) + 1
        state["locked_step_block_counts"] = counts
        blocked_reason = (
            f"Step {step_number} is locked after a successful validation/execution and cannot be changed."
        )
    elif planned_action and planned_action != action:
        blocked_reason = (
            f"Step {step_number} is planned as '{planned_action}', not '{action}'. "
            "Execute the planned action for this step or produce a new plan."
        )
    elif prev_action and prev_action != action:
        blocked_reason = (
            f"Step {step_number} is already registered as action '{prev_action}' "
            f"and cannot be changed to '{action}'."
        )

    def _norm(s: str) -> str:
        return " ".join((s or "").split())

    if blocked_reason is None and name == "tool_execute_query_step":
        valid_sql_flag = step_obj.get("validSql")
        validated_entry = validated_sqls.get(_validated_sql_key(run_id, step_number))

        if action != "query":
            blocked_reason = "step.action must be 'query'."
        elif step_number is None:
            blocked_reason = "step.step is required."
        elif not database:
            blocked_reason = "step.database is required."
        elif not sql:
            blocked_reason = "step.sql is required."
        elif planned_database and planned_database != database:
            blocked_reason = (
                f"Step {step_number} is planned to query database '{planned_database}', "
                f"not '{database}'. Execute the planned query for this step or produce a new plan."
            )
        elif prev_action == "query" and prev_database and prev_database != database:
            blocked_reason = (
                f"Step {step_number} is already registered for database '{prev_database}' "
                f"and cannot be changed to '{database}'."
            )
        elif validated_entry is None:
            blocked_reason = (
                f"No validated SQL found for step {step_number}. "
                "Call tool_validate_sql first with this step_number, "
                "then call tool_execute_query_step with the identical SQL."
            )
        elif _norm(validated_entry.get("sql", "")) != _norm(sql):
            blocked_reason = (
                "SQL does not match the validated entry — execute the exact same SQL. "
                f"Validated SQL starts with: {validated_entry.get('sql', '')[:120]!r}"
            )
        elif validated_entry.get("database", "") != database:
            blocked_reason = (
                f"Database mismatch: validated for '{validated_entry.get('database')}' "
                f"but received '{database}'."
            )
    elif blocked_reason is None:
        max_retries = getattr(state.get("params"), "max_step_retries", 3)
        if step_number is not None and not _step_retry_ok(state, step_number, max_retries):
            state["replan_trigger"] = {
                "reason": "blocked_step",
                "error": f"Step {step_number} exhausted {action} retry budget ({max_retries} attempts).",
                "zero_rows_step": None,
            }
            blocked_reason = (
                f"Step {step_number} has exhausted its {action} retry budget ({max_retries} attempts). "
                "Produce a new plan."
            )
        elif step_number is None:
            blocked_reason = "step.step is required."
        elif action == "aggregate":
            if step_obj.get("source_step") is None:
                counts = dict(state.get("missing_source_step_counts", {}))
                counts[action] = counts.get(action, 0) + 1
                state["missing_source_step_counts"] = counts
                blocked_reason = "aggregate requires source_step."
            elif not step_obj.get("group_by") and not step_obj.get("aggregation_type"):
                agg_list = (
                    step_obj.get("aggregate")
                    or step_obj.get("aggregates")
                    or step_obj.get("aggregations")
                )
                has_raw_sql = isinstance(agg_list, list) and any(
                    isinstance(a, dict) and "aggregate_sql" in a for a in agg_list
                )
                if has_raw_sql:
                    blocked_reason = (
                        "aggregate does not support 'aggregate_sql'. "
                        "Use aggregation_type (e.g. 'count', 'sum', 'avg', 'min', 'max') "
                        "and optionally group_by=['column'] and metric_column='col'. "
                        "For complex spatial calculations, use tool_execute_query_step instead."
                    )
                elif agg_list:
                    blocked_reason = None
                else:
                    blocked_reason = (
                        "aggregate requires group_by or aggregation_type. "
                        "Example: {\"action\": \"aggregate\", \"source_step\": N, "
                        "\"group_by\": [\"kommunenavn\"], \"aggregation_type\": \"count\", \"metric_as\": \"building_count\"}"
                    )
        elif action in SOURCE_STEP_ACTIONS and step_obj.get("source_step") is None:
            counts = dict(state.get("missing_source_step_counts", {}))
            counts[action] = counts.get(action, 0) + 1
            state["missing_source_step_counts"] = counts
            blocked_reason = f"{action} requires source_step."
        elif action in DUAL_SOURCE_ACTIONS:
            left = step_obj.get("left_source")
            right = step_obj.get("right_source")
            if left is None or right is None:
                counts = dict(state.get("missing_source_step_counts", {}))
                counts[action] = counts.get(action, 0) + 1
                state["missing_source_step_counts"] = counts
                blocked_reason = f"{action} requires left_source and right_source as plain integers (step numbers)."
            elif not isinstance(left, int) or not isinstance(right, int):
                blocked_reason = (
                    f"{action}: left_source and right_source must be plain integers (step numbers), "
                    f"not {type(left).__name__}/{type(right).__name__}. "
                    f"Got left_source={left!r}, right_source={right!r}."
                )
            elif action == "spatial_join":
                valid_join_types = {"intersects", "within", "contains", "dwithin", "touches", "overlaps"}
                jt = step_obj.get("join_type")
                if jt not in valid_join_types:
                    blocked_reason = (
                        f"spatial_join: join_type must be a spatial predicate — one of "
                        f"{sorted(valid_join_types)}. Got {jt!r}. "
                        f"Do NOT use SQL join types like 'inner', 'left', 'outer'."
                    )

    if blocked_reason:
        return _block_step_result(name, blocked_reason, step_number, action), None

    return None, step_obj





















async def _execute_tool(
    tool: Any,
    name: str,
    args: dict,
    blocked_json: str | None,
) -> tuple[str, Any, bool, int]:
    """
    Either return the pre-built blocked result or invoke the tool.

    Returns:
        (result_str, parsed_result, exec_ok, rows_returned)
    """
    if blocked_json is not None:
        parsed = _parse_tool_result(blocked_json)
        return blocked_json, parsed, False, 0

    step_obj = args.get("step") if isinstance(args.get("step"), dict) else {}
    step_number = step_obj.get("step") if isinstance(step_obj, dict) else None
    logger.info("      [execute] STARTING %s step=%s", name, step_number)

    result_str = await tool.ainvoke(args)
    parsed = _parse_tool_result(result_str)
    exec_ok = isinstance(parsed, dict) and parsed.get("executed", False)
    rows_returned = parsed.get("rows_returned", 0) if isinstance(parsed, dict) else 0
    logger.info(
        "      [execute] FINISHED %s step=%s ok=%s rows=%s",
        name, step_number, exec_ok, rows_returned,
    )
    return result_str, parsed, exec_ok, rows_returned














def _update_step_registry(
    state: "AgentState",
    name: str,
    step_obj: dict | None,
    exec_ok: bool,
    rows_returned: int,
) -> None:
    """Write step execution metadata into state['step_registry']."""
    if step_obj is None:
        return

    step_number = step_obj.get("step")
    if step_number is None:
        return

    action = step_obj.get("action") or name.replace("tool_execute_", "").replace("_step", "")
    is_query = name == "tool_execute_query_step"

    step_registry = dict(state.get("step_registry", {}))
    prev = dict(step_registry.get(step_number, {}))
    if prev.get("locked"):
        return

    prev.update({
        "step": step_number,
        "action": action,
        "sql": (step_obj.get("sql") or "").strip() if is_query else prev.get("sql", ""),
        "database": (step_obj.get("database") or "").strip() if is_query else prev.get("database", ""),
        "validated": prev.get("validated", False),
        "executed": exec_ok,
        "rows_returned": rows_returned,
        "status": "empty" if exec_ok and rows_returned == 0 else ("ok" if exec_ok else "error"),
        "empty_result": bool(exec_ok and rows_returned == 0),
        "terminal_empty_result": bool(
            exec_ok
            and rows_returned == 0
            and _is_terminal_required_step(state, step_number)
        ),
        "locked": prev.get("locked", False),
        "step_payload": {k: v for k, v in step_obj.items() if k not in ("step", "action", "sql", "database")},
    })
    step_registry[step_number] = prev
    state["step_registry"] = step_registry




























def _update_iteration_flags(
    state: "AgentState",
    name: str,
    step_obj: dict | None,
    parsed_result: Any,
    exec_ok: bool,
    iter_flags: dict,
) -> None:
    step_number = step_obj.get("step") if isinstance(step_obj, dict) else None
    if step_number is None and isinstance(parsed_result, dict):
        step_number = parsed_result.get("step")
    run_id = state.get("run_id")
    rows_returned = 0
    if isinstance(parsed_result, dict):
        try:
            rows_returned = int(parsed_result.get("rows_returned") or parsed_result.get("row_count") or 0)
        except (TypeError, ValueError):
            rows_returned = 0

    if exec_ok:
        iter_flags["iter_any_exec_ok"] = True
        state["last_failed_step"] = None
        state["last_guard_error"] = None
        if rows_returned == 0:
            state["last_empty_step"] = step_number
        else:
            state["last_empty_step"] = None

        if not _is_intermediate_empty_step(state, step_number, rows_returned):
            _lock_step(state, step_number)

        if name == "tool_execute_query_step":
            validated_sqls = dict(state.get("validated_sqls", {}))
            key = _validated_sql_key(run_id, step_number)
            if key:
                validated_sqls.pop(key, None)
            state["validated_sqls"] = validated_sqls
    else:
        iter_flags["iter_had_error"] = True
        state["last_failed_step"] = step_number
        state["last_guard_error"] = None
        state["last_execute_error"] = json.dumps(parsed_result, ensure_ascii=False)

        is_blocked = isinstance(parsed_result, dict) and str(
            parsed_result.get("error", "")
        ).endswith("blocked.")
        if is_blocked:
            attempted_action = None
            if isinstance(step_obj, dict):
                attempted_action = step_obj.get("action")
            if not attempted_action and isinstance(parsed_result, dict):
                attempted_action = parsed_result.get("action")
            if not attempted_action:
                attempted_action = name.replace("tool_execute_", "").replace("_step", "")
            attempted_database = (step_obj.get("database") or "").strip() if isinstance(step_obj, dict) else ""
            replan_payload = _plan_mismatch_replan_payload(
                state,
                step_number,
                attempted_action=attempted_action,
                attempted_database=attempted_database,
            )
            if replan_payload is not None:
                state["replan_trigger"] = replan_payload
        else:
            if name == "tool_execute_query_step":
                _increment_sql_repair(state, step_number)
                state["sql_repair_count"] = state.get("sql_repair_count", 0) + 1
                state["total_sql_repairs"] = state.get("total_sql_repairs", 0) + 1
            else:
                _increment_step_retry(state, step_number)
                state["step_retry_count"] = state.get("step_retry_count", 0) + 1
                state["total_step_retries"] = state.get("total_step_retries", 0) + 1

        state["all_errors"] = _dedup(
            list(state.get("all_errors", [])) + [json.dumps(parsed_result, ensure_ascii=False)]
        )

    if exec_ok:
        state["last_executed_result"] = parsed_result
        state["last_execute_error"] = None


















def _handle_validate_sql_tool(
    state: "AgentState",
    args: dict,
    result: str,
    iter_flags: dict,
) -> None:
    sql = (args.get("sql") or "").strip()
    step_number = args.get("step_number")
    database = (args.get("databaseName") or "").strip()
    run_id = state.get("run_id")

    parsed_result = _parse_tool_result(result)
    ok = isinstance(parsed_result, dict) and parsed_result.get("validated") is True
    is_blocked = isinstance(parsed_result, dict) and str(
        parsed_result.get("error", "")
    ).endswith("blocked.")
    validated_sqls = dict(state.get("validated_sqls", {}))
    validated_key = _validated_sql_key(run_id, step_number)
    had_validated_entry = bool(validated_key and validated_key in validated_sqls)
    step_registry = dict(state.get("step_registry", {}))
    prev = dict(step_registry.get(step_number, {})) if step_number is not None else {}
    prev_payload = prev.get("step_payload") if isinstance(prev.get("step_payload"), dict) else {}
    prev_action = prev_payload.get("action") or prev.get("action")
    is_locked = bool(prev.get("locked")) or step_number in set(state.get("locked_step_ids", set()))

    state["last_sql"] = sql
    state["last_validation_ok"] = ok
    state["last_validation_error"] = None if (ok or is_blocked) else json.dumps(parsed_result, ensure_ascii=False)
    state["last_validation_invalidated"] = False
    state["last_validation_invalidated_reason"] = None

    if is_blocked:
        iter_flags["iter_had_error"] = True
        state["last_failed_step"] = step_number
        state["last_validated_sql"] = None
        if validated_key:
            validated_sqls.pop(validated_key, None)
            state["validated_sqls"] = validated_sqls
        if had_validated_entry:
            state["last_validation_invalidated"] = True
            state["last_validation_invalidated_reason"] = "blocked"
        state["last_guard_error"] = json.dumps(parsed_result, ensure_ascii=False)
        state["all_errors"] = _dedup(
            list(state.get("all_errors", [])) + [json.dumps(parsed_result, ensure_ascii=False)]
        )
        replan_payload = _plan_mismatch_replan_payload(
            state,
            step_number,
            attempted_action="query",
            attempted_database=database,
        )
        if replan_payload is not None:
            state["replan_trigger"] = replan_payload
        return

    if step_number is not None and not is_locked:
        prev.update({
            "step": step_number,
            "action": prev_action or "query",
            "sql": sql,
            "database": database,
            "validated": ok,
            "executed": prev.get("executed", False),
            "rows_returned": prev.get("rows_returned", 0),
            "status": prev.get("status", "pending"),
            "locked": prev.get("locked", False),
            "step_payload": {},
        })
        step_registry[step_number] = prev
        state["step_registry"] = step_registry

    if ok:
        state["last_validated_sql"] = sql
        state["last_guard_error"] = None
        if validated_key:
            validated_sqls[validated_key] = {"sql": sql, "database": database}
        state["validated_sqls"] = validated_sqls


    else:
        iter_flags["iter_had_validation_error"] = True
        iter_flags["iter_had_error"] = True
        state["last_failed_step"] = step_number
        state["last_validated_sql"] = None
        if validated_key:
            validated_sqls.pop(validated_key, None)
            state["validated_sqls"] = validated_sqls
        if had_validated_entry:
            state["last_validation_invalidated"] = True
            state["last_validation_invalidated_reason"] = "failed"
        state["last_guard_error"] = None
        
        _increment_sql_repair(state, step_number)
    
        state["sql_repair_count"] = state.get("sql_repair_count", 0) + 1

        state["total_sql_repairs"] = state.get("total_sql_repairs", 0) + 1
        state["all_errors"] = _dedup(
            list(state.get("all_errors", [])) + [json.dumps(parsed_result, ensure_ascii=False)]
        )

















def _handle_discovery_tool(
    state: "AgentState",
    name: str,
    args: dict,
    result: str,
) -> None:
    parsed = _parse_tool_result(result)

    if name == "tool_get_database_catalog":
        dbs, tables = _extract_from_database_catalog(parsed)
        state["discovered_databases"] = _dedup(list(state.get("discovered_databases", [])) + dbs)
        state["discovered_tables"] = _dedup(list(state.get("discovered_tables", [])) + tables)

    elif name == "tool_get_available_databases":
        pass  # listing available databases is not the same as exploring them

    elif name == "tool_get_available_tables":
        tables = _extract_tables_from_available_tables_result(parsed)
        state["discovered_tables"] = _dedup(list(state.get("discovered_tables", [])) + tables)

    elif name == "tool_get_table_schema":
        table = args.get("table_name")
        if table:
            state["discovered_tables"] = _dedup(list(state.get("discovered_tables", [])) + [table])


















def _build_tool_message_and_log(
    tc: dict,
    i: int,
    name: str,
    raw_args: dict,
    args: dict,
    result: Any,
    latency_s: float | None,
    msgs: list,
    iteration_tools: list,
) -> None:
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)

    iteration_tools.append({
        "tool": name,
        "raw_args": raw_args,
        "args": args,
        "latency_s": latency_s,
        "result": content,
    })

    logger.info("      [tool] TOOL   : %s", name)
    logger.info("      [tool] ARGS   : %s", args)
    logger.info("      [tool] RESULT : %s", content)
    if latency_s is not None:
        logger.info("      [tool] LAT    : %.6fs", latency_s)

    tool_call_id = tc.get("id") or f"fallback_{i}"
    msgs.append(ToolMessage(content=content, tool_call_id=tool_call_id))


def _build_step_event(
    state: "AgentState",
    name: str,
    raw_args: dict,
    args: dict,
    result: Any,
    latency_s: float | None = None,
) -> dict | None:
    parsed_result = _parse_tool_result(result)

    if name == "tool_validate_sql":
        step_number = args.get("step_number")
        if step_number is None:
            return None

        step_meta = dict(state.get("step_registry", {}).get(step_number, {}))
        return {
            "event_type": "validate",
            "step": step_number,
            "action": step_meta.get("action", "query"),
            "database": (args.get("databaseName") or "").strip() or step_meta.get("database"),
            "sql": (args.get("sql") or "").strip() or step_meta.get("sql"),
            "validated": step_meta.get("validated", False),
            "executed": step_meta.get("executed", False),
            "status": step_meta.get("status"),
            "rows_returned": step_meta.get("rows_returned"),
            "sql_repairs": state.get("sql_repair_counts", {}).get(step_number, 0),
            "step_retries": state.get("step_retry_counts", {}).get(step_number, 0),
            "validation_invalidated": bool(state.get("last_validation_invalidated", False)),
            "validation_invalidated_reason": state.get("last_validation_invalidated_reason"),
            "latency_s": latency_s,
        }

    if name in NON_DISCOVERY_EXEC_TOOLS:
        action = "route" if name == "tool_get_route" else name.replace("tool_", "")
        return {
            "event_type": "execute",
            "step": None,
            "action": action,
            "database": None,
            "sql": None,
            "validated": True,
            "executed": isinstance(parsed_result, dict) and parsed_result.get("executed", False),
            "status": "ok" if (isinstance(parsed_result, dict) and parsed_result.get("executed", False)) else "error",
            "rows_returned": int(parsed_result.get("rows_returned") or 0) if isinstance(parsed_result, dict) else 0,
            "sql_repairs": 0,
            "step_retries": 0,
            "latency_s": latency_s,
        }

    if name not in STEP_TOOLS:
        return None

    step_obj = args.get("step")
    step_number = step_obj.get("step") if isinstance(step_obj, dict) else None
    if step_number is None:
        return None

    step_meta = dict(state.get("step_registry", {}).get(step_number, {}))
    return {
        "event_type": "execute",
        "step": step_number,
        "action": step_meta.get("action") or name.replace("tool_execute_", "").replace("_step", ""),
        "database": step_meta.get("database"),
        "sql": step_meta.get("sql"),
        "validated": step_meta.get("validated", False),
        "executed": step_meta.get("executed", False),
        "status": step_meta.get("status"),
        "rows_returned": step_meta.get("rows_returned"),
        "sql_repairs": state.get("sql_repair_counts", {}).get(step_number, 0),
        "step_retries": state.get("step_retry_counts", {}).get(step_number, 0),
        "latency_s": latency_s,
    }

















def _get_tools():
    return [
        tool_get_available_tables, 
        tool_get_available_databases,
        tool_get_table_schema,
        tool_get_database_catalog,
        tool_validate_sql,
        

        tool_execute_query_step,
        tool_execute_filter_step,
        tool_execute_select_columns_step,
        tool_execute_spatial_join_step,
        tool_execute_attribute_join_step,
        tool_execute_buffer_step,
        tool_execute_nearest_step,
        tool_execute_distance_step,
        tool_execute_aggregate_step,



        tool_execute_sort_step,
        tool_execute_limit_step,
        tool_execute_clip_step,
        tool_execute_union_step,
        tool_execute_intersection_step,
        tool_execute_difference_step,


        tool_execute_dissolve_step,
        tool_execute_centroid_step,
        tool_execute_transform_crs_step,
    ]




async def _tools_node(state: AgentState) -> AgentState:
    tools = _get_tools()
    tool_map = {t.name: t for t in tools}

    ai = state["current_ai"]
    msgs = list(state["msgs"])
    iteration_tools: list[dict] = []
    iteration_step_events: list[dict] = []

    iter_flags: dict = {
        "iter_had_error": False,
        "iter_had_validation_error": False,
        "iter_any_exec_ok": False,
    }

    for i, tc in enumerate(ai.tool_calls, start=1):
        name = tc["name"]
        raw_args = _coerce_args(tc.get("args"))


        args = _normalize_tool_args(name, raw_args, state["run_id"])
        tool = tool_map.get(name)
        state["total_tool_calls"] = state.get("total_tool_calls", 0) + 1
        tool_latency_s: float | None = None

        if tool is None:
            err = f"Unknown tool: {name}"
            iter_flags["iter_had_error"] = True

            state["all_errors"] = _dedup(list(state.get("all_errors", [])) + [err])
            _build_tool_message_and_log(tc, i, name, raw_args, args, err, 0.0, msgs, iteration_tools)
            

            


            continue

        result_str = ""
        tool_t0 = time.perf_counter()

        try:
            if name in STEP_TOOLS:
                validated_sqls = state.get("validated_sqls", {})
                attempted_step = (
                    dict(args.get("step"))
                    if isinstance(args.get("step"), dict)
                    else None
                )
                blocked_json, step_obj = _guard_step_tool(state, name, raw_args, args, validated_sqls)

                result_str, parsed_result, exec_ok, rows_returned = await _execute_tool(
                    tool, name, args, blocked_json
                )

                _update_step_registry(state, name, step_obj, exec_ok, rows_returned)
                _update_iteration_flags(
                    state,
                    name,
                    step_obj if step_obj is not None else attempted_step,
                    parsed_result,
                    exec_ok,
                    iter_flags,
                )

                # A successful step with 0 rows is still a valid executed result.
                # Do not auto-trigger a backend replan here.
                if (
                    exec_ok
                    and rows_returned == 0
                    and state.get("replan_trigger") is None
                    and state.get("plan_finalized")
                ):
                    required = {str(s) for s in state.get("required_step_ids", set())}
                    executed_now = {
                        str(k) for k, v in state.get("step_registry", {}).items()
                        if v.get("executed")
                    }
                    remaining = required - executed_now
                    step_number = step_obj.get("step") if isinstance(step_obj, dict) else None
                    if _is_terminal_required_step(state, step_number):
                        logger.info(
                            "      [zero-rows] Step %s returned 0 rows as terminal result — accepting as empty result",
                            step_number,
                        )
                    else:
                        sig = _zero_rows_sig(step_obj or {})
                        counts = dict(state.get("zero_rows_replan_counts", {}))
                        counts[sig] = counts.get(sig, 0) + 1
                        state["zero_rows_replan_counts"] = counts
                        state["replan_trigger"] = {
                            "reason": "zero_rows_step",
                            "error": (
                                f"Step {step_number} returned 0 rows before the plan was complete. "
                                "Treat this as an empty intermediate result and revise the plan, "
                                "filter, join, table choice, or predicate."
                            ),
                            "zero_rows_step": step_number,
                            "zero_rows_sig": sig,
                            "empty_count": counts[sig],
                            "remaining_required_steps": sorted(remaining),
                        }
                        logger.info(
                            "      [zero-rows] Step %s returned 0 rows before completion — replan required (%s remaining planned steps)",
                            step_number, len(remaining),
                        )

            elif name == "tool_validate_sql":
                blocked_json = _guard_validate_sql(state, args)
                if blocked_json is not None:
                    result_str = blocked_json
                else:
                    result_str = await tool.ainvoke(args)
                _handle_validate_sql_tool(state, args, result_str, iter_flags)

            elif name in NON_DISCOVERY_EXEC_TOOLS:
                result_str = await tool.ainvoke(args)
                parsed_result = _parse_tool_result(result_str)
                exec_ok = isinstance(parsed_result, dict) and parsed_result.get("executed", False)
                if exec_ok:
                    iter_flags["iter_any_exec_ok"] = True
                    state["last_executed_result"] = parsed_result
                else:
                    iter_flags["iter_had_error"] = True
                    state["all_errors"] = _dedup(
                        list(state.get("all_errors", [])) + [json.dumps(parsed_result, ensure_ascii=False)]
                    )

            else:
                result_str = await tool.ainvoke(args)
                _handle_discovery_tool(state, name, args, result_str)
            tool_latency_s = round(time.perf_counter() - tool_t0, 6)

            _build_tool_message_and_log(
                tc, i, name, raw_args, args, result_str, tool_latency_s, msgs, iteration_tools
            )
            step_event = _build_step_event(
                state, name, raw_args, args, result_str, tool_latency_s
            )
            if step_event is not None:
                iteration_step_events.append(step_event)

        except Exception as e:
            err = f"Error: {e}"
            iter_flags["iter_had_error"] = True
            state["all_errors"] = _dedup(list(state.get("all_errors", [])) + [err])
            tool_latency_s = round(time.perf_counter() - tool_t0, 6)
            _build_tool_message_and_log(
                tc, i, name, raw_args, args, err, tool_latency_s, msgs, iteration_tools
            )
            step_event = _build_step_event(
                state, name, raw_args, args, err, tool_latency_s
            )
            if step_event is not None:
                iteration_step_events.append(step_event)

    state["iter_had_error"] = iter_flags["iter_had_error"]
    state["iter_had_validation_error"] = iter_flags["iter_had_validation_error"]
    state["iter_any_exec_ok"] = iter_flags["iter_any_exec_ok"]
    state["last_execute_ok"] = iter_flags["iter_any_exec_ok"]

    run_log = list(state.get("current_run_log", []))
    if run_log:
        run_log[-1]["tools"] = iteration_tools
        run_log[-1]["step_events"] = iteration_step_events
        tool_latency_total = round(
            sum(float(t.get("latency_s") or 0.0) for t in iteration_tools),
            6,
        )
        run_log[-1]["tool_latency_s"] = tool_latency_total
        run_log[-1]["iteration_latency_s"] = round(
            float(run_log[-1].get("model_latency_s") or 0.0) + tool_latency_total,
            6,
        )
        run_log[-1]["iteration_error"] = (
            state.get("last_guard_error")
            or state.get("last_execute_error")
            or state.get("last_validation_error")
        )
        run_log[-1]["failed_step"] = state.get("last_failed_step")
        run_log[-1]["had_error"] = iter_flags["iter_had_error"]
        run_log[-1]["had_validation_error"] = iter_flags["iter_had_validation_error"]
        run_log[-1]["had_exec_success"] = iter_flags["iter_any_exec_ok"]

    state["msgs"] = msgs
    state["current_run_log"] = run_log
    state["crash_snapshot"] = _build_snapshot(state)
    return state
