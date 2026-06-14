import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from agent_models import TokenUsage, AgentState

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency fallback
    tiktoken = None



import logging

logger = logging.getLogger(__name__)
_ESTIMATED_TOKENS_ENCODING_NAME = "o200k_base"
_ESTIMATED_TOKENS_ENCODING = None


_PLAN_ACTIONS = (
    "spatial_join",
    "attribute_join",
    "select_columns",
    "transform_crs",
    "intersection",
    "difference",
    "aggregate",
    "dissolve",
    "centroid",
    "distance",
    "nearest",
    "buffer",
    "filter",
    "query",
    "sort",
    "limit",
    "clip",
    "union",
)


# def load_prompt(name: str) -> str:
#     path = Path(__file__).parent / "prompts" / f"{name}.txt"
#     if not path.exists():
#         raise ValueError(f"Prompt not found: {path}")
#     return path.read_text(encoding="utf-8")



def load_prompt(name: str) -> str:
    path = Path(__file__).parent / f"{name}.txt"
    if not path.exists():
        raise ValueError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")




def _short(s: str, n: int = 120) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _dedup(xs: list[str]) -> list[str]:
    seen, out = set(), []
    for x in xs or []:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _empty_token_dict() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _add_tokens(total: dict, inc: TokenUsage) -> dict:
    total["prompt_tokens"] += inc.prompt_tokens
    total["completion_tokens"] += inc.completion_tokens
    total["total_tokens"] += inc.total_tokens
    return total


def _extract_tokens(ai) -> TokenUsage:
    usage = getattr(ai, "usage_metadata", None) or getattr(
        ai, "response_metadata", {}
    ).get("token_usage", {})
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        ct = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        tt = usage.get("total_tokens", 0) or (pt + ct)
        return TokenUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)
    pt = getattr(usage, "input_tokens", 0)
    ct = getattr(usage, "output_tokens", 0)
    tt = getattr(usage, "total_tokens", 0) or (pt + ct)
    return TokenUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0

    global _ESTIMATED_TOKENS_ENCODING
    if tiktoken is not None:
        try:
            if _ESTIMATED_TOKENS_ENCODING is None:
                _ESTIMATED_TOKENS_ENCODING = tiktoken.get_encoding(
                    _ESTIMATED_TOKENS_ENCODING_NAME
                )
            return len(_ESTIMATED_TOKENS_ENCODING.encode(text))
        except Exception:
            pass

    return len(re.findall(r"\S+", text))


def _json_for_token_estimate(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _serialize_message_for_token_estimate(msg: Any) -> str:
    payload = {
        "type": getattr(msg, "type", msg.__class__.__name__),
        "name": getattr(msg, "name", None),
        "tool_call_id": getattr(msg, "tool_call_id", None),
        "content": getattr(msg, "content", None),
        "tool_calls": getattr(msg, "tool_calls", None),
        "invalid_tool_calls": getattr(msg, "invalid_tool_calls", None),
        "additional_kwargs": getattr(msg, "additional_kwargs", None),
    }
    return _json_for_token_estimate(payload)


def _estimate_messages_tokens(messages: list[Any] | None) -> int:
    total = 0
    for msg in messages or []:
        total += _estimate_text_tokens(_serialize_message_for_token_estimate(msg))
    return total


def _estimate_ai_tokens(ai: Any) -> int:
    payload = {
        "content": getattr(ai, "content", None),
        "tool_calls": getattr(ai, "tool_calls", None),
        "invalid_tool_calls": getattr(ai, "invalid_tool_calls", None),
        "additional_kwargs": getattr(ai, "additional_kwargs", None),
    }
    return _estimate_text_tokens(_json_for_token_estimate(payload))


def _estimate_exchange_tokens(messages: list[Any] | None, ai: Any) -> TokenUsage:
    prompt_tokens = _estimate_messages_tokens(messages)
    completion_tokens = _estimate_ai_tokens(ai)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _parse_tool_result(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {"executed": False, "error": raw}
    return raw


def _extract_tables_from_available_tables_result(parsed: Any) -> list[str]:
    if isinstance(parsed, dict):
        if isinstance(parsed.get("tables"), list):
            return [t for t in parsed["tables"] if isinstance(t, str)]
        if isinstance(parsed.get("data"), list):
            return [t for t in parsed["data"] if isinstance(t, str)]
    if isinstance(parsed, list):
        return [t for t in parsed if isinstance(t, str)]
    return []


def _extract_databases_from_result(parsed: Any) -> list[str]:
    def _name(x: Any) -> str | None:
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get("database") or x.get("name") or None
        return None

    if isinstance(parsed, list):
        return [n for x in parsed if (n := _name(x))]
    if isinstance(parsed, dict):
        for key in ("databases", "data"):
            items = parsed.get(key)
            if isinstance(items, list):
                return [n for x in items if (n := _name(x))]
    return []


# def _format_restart_summary(state: "AgentState") -> str:
#     return f"""
# Original query:
# {state["user_input"]}
#
# Discovered databases:
# {state.get("discovered_databases", [])}
#
# Discovered tables:
# {state.get("discovered_tables", [])}
#
# Last failed SQL:
# {state.get("last_sql")}
#
# Last validation error:
# {state.get("last_validation_error")}
#
# Last guard error:
# {state.get("last_guard_error")}
#
# Last execution error:
# {state.get("last_execute_error")}
#
# Do not repeat the same failed SQL.
# Avoid redundant metadata calls if the needed metadata has already been discovered.
# """.strip()


def _sorted_step_entries(steps: dict | None) -> list[tuple]:
    return sorted(
        (steps or {}).items(),
        key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]),
    )


def _normalize_step_key(step_key: Any) -> str:
    return str(step_key)


def _normalize_step_mapping(mapping: dict | None) -> dict:
    return {
        _normalize_step_key(k): v
        for k, v in (mapping or {}).items()
    }






def _extract_plan_from_text(text: str | None) -> tuple[dict, set]:
    raw = (text or "").strip()
    if not raw:
        return {}, set()

    # Try JSON first — extract outermost JSON object to handle mixed text (prefix + JSON + suffix)
    def _parse_plan_json(text: str) -> dict | None:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("planned_steps"), list):
                return parsed
        except Exception:
            pass
        import re as _re
        m = _re.search(r'\{.*\}', text, _re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, dict) and isinstance(parsed.get("planned_steps"), list):
                    return parsed
            except Exception:
                pass
        return None

    parsed = _parse_plan_json(raw)
    if parsed is None or not isinstance(parsed.get("planned_steps"), list):
        return {}, set()

    planned_steps: dict = {}
    for item in parsed["planned_steps"]:
        if not isinstance(item, dict):
            continue
        step_no = item.get("step")
        action = item.get("action")
        if not isinstance(step_no, int) or not isinstance(action, str):
            continue
        planned_steps[step_no] = {
            "step": step_no,
            "action": action,
            "description": item.get("purpose", ""),
            "database": item.get("database"),
        }

    required_step_ids = set(planned_steps.keys())
    return planned_steps, required_step_ids














def _collect_step_stats(run_logs: list | None) -> tuple[dict, dict, dict]:
    step_registry = {}
    sql_repairs_by_step = {}
    step_retries_by_step = {}

    for run in (run_logs or []):
        step_registry.update(_normalize_step_mapping(run.get("step_registry", {})))
        for step_no, count in (run.get("sql_repair_counts") or {}).items():
            sql_repairs_by_step[_normalize_step_key(step_no)] = count
        for step_no, count in (run.get("step_retry_counts") or {}).items():
            step_retries_by_step[_normalize_step_key(step_no)] = count

    return step_registry, sql_repairs_by_step, step_retries_by_step


def _active_plan_run(run_logs: list | None) -> dict | None:
    """Return the run entry that represents the active plan.

    After replans, step registries are renumbered from 1 again. Any step-based
    reporting must therefore read from the latest run only, not from an older
    finalized plan or a merge of historical runs.
    """
    runs = [run for run in (run_logs or []) if isinstance(run, dict)]
    if not runs:
        return None

    return runs[-1]


def _active_plan_logs(run_logs: list | None) -> list[dict]:
    run = _active_plan_run(run_logs)
    return [run] if run else []


def _collect_active_step_stats(run_logs: list | None) -> tuple[dict, dict, dict]:
    return _collect_step_stats(_active_plan_logs(run_logs))


def _last_run(run_logs: list | None) -> list:
    """Return only the last run entry as a single-element list."""
    runs = run_logs or []
    return [runs[-1]] if runs else []


def _build_step_outcomes(run_logs: list | None) -> list[dict]:
    step_registry, _, _ = _collect_active_step_stats(run_logs)
    outcomes: list[dict] = []

    for step_no, meta in _sorted_step_entries(step_registry):
        outcomes.append(
            {
                "step": step_no,
                "action": meta.get("action"),
                "validated": meta.get("validated", False),
                "executed": meta.get("executed", False),
                "rows_returned": meta.get("rows_returned"),
                "status": meta.get("status"),
            }
        )

    return outcomes


def _valid_sql_from_steps(
    run_logs: list | None,
    valid_sql_fallback: bool = False,
) -> bool:
    outcomes = _build_step_outcomes(run_logs)
    if not outcomes:
        return valid_sql_fallback

    query_steps = [step for step in outcomes if step.get("action") == "query"]
    if query_steps:
        return all(step.get("validated", False) for step in query_steps)

    return all(
        step.get("executed", False) and step.get("status") in {"ok", "empty"}
        for step in outcomes
    )


def _executed_from_steps(
    run_logs: list | None,
    executed_fallback: bool = False,
) -> bool:
    outcomes = _build_step_outcomes(run_logs)
    if not outcomes:
        return executed_fallback
    return any(step.get("executed", False) for step in outcomes)


def _step_execution_map(run_logs: list | None) -> dict[str, bool]:
    outcomes = _build_step_outcomes(run_logs)
    return {str(s["step"]): bool(s.get("executed", False)) for s in outcomes}


def _format_step_execution(step_map: dict[str, bool]) -> str:
    if not step_map:
        return "-"
    parts = []
    for step_no in sorted(step_map, key=lambda x: int(x) if x.isdigit() else x):
        parts.append(f"{step_no}:{'T' if step_map[step_no] else 'F'}")
    return " ".join(parts)


def _terminal_required_step_ids(required_ids: set[str]) -> set[str]:
    if not required_ids:
        return set()

    numeric_ids = [int(step_id) for step_id in required_ids if str(step_id).isdigit()]
    if numeric_ids and len(numeric_ids) == len(required_ids):
        return {str(max(numeric_ids))}

    return {sorted(str(step_id) for step_id in required_ids)[-1]}


def _step_counts_as_completed(step: dict, terminal_step_ids: set[str]) -> bool:
    if not step.get("executed", False):
        return False

    status = step.get("status")
    if status == "ok":
        return True

    step_id = _normalize_step_key(step.get("step"))
    return status == "empty" and step_id in terminal_step_ids


def _answer_indicates_incomplete_task(answer: str | None) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return False

    incomplete_markers = (
        "can proceed to the next step",
        "can proceed to next step",
        "can proceed to the next steps",
        "can proceed to next steps",
        "next step would be",
        "next steps would be",
        "next i would",
        "i can now proceed",
        "i can proceed",
        "we can proceed",
        "the next step is",
        "the next steps are",
        "still need to",
        "need to run",
        "need to query",
        "need to perform",
        "then run",
        "then perform the spatial join",
        "hazard query still",
        "have not yet",
        "haven't yet",
        "not yet run",
        "not yet performed",
    )
    return any(marker in text for marker in incomplete_markers)


def _task_completed_from_steps(
    run_logs: list | None,
    task_completed_fallback: bool = False,
    final_answer: str | None = None,
    required_step_ids: set | list | None = None,
    plan_finalized: bool = False,
    planned_steps: dict | None = None,
    original_required_step_ids: set | list | None = None,
) -> bool:
    required_ids = {
        _normalize_step_key(step_id)
        for step_id in (required_step_ids or [])
    }
    original_ids = {
        _normalize_step_key(step_id)
        for step_id in (original_required_step_ids or [])
    }
    outcomes = _build_step_outcomes(run_logs)
    if not outcomes:
        return (
            task_completed_fallback
            and not required_ids
            and not plan_finalized
            and not _answer_indicates_incomplete_task(final_answer)
        )

    if plan_finalized and not required_ids:
        return False

    if not required_ids:
        return False

    outcome_by_step = {
        _normalize_step_key(step.get("step")): step
        for step in outcomes
        if step.get("step") is not None
    }

    if not outcome_by_step:
        return False

    terminal_step_ids = _terminal_required_step_ids(required_ids)

    steps_ok = all(
        _step_counts_as_completed(step, terminal_step_ids)
        for step in outcomes
        if _normalize_step_key(step.get("step")) in required_ids
    )
    if not steps_ok:
        return False

    if planned_steps and required_ids:
        ok_step_ids = {
            k for k, v in outcome_by_step.items()
            if _step_counts_as_completed(v, terminal_step_ids)
        }
        query_required_ids = {
            _normalize_step_key(step_id)
            for step_id, step_meta in planned_steps.items()
            if step_meta.get("action") == "query"
            and _normalize_step_key(step_id) in required_ids
        }
        # Required non-query actions (spatial_join, aggregate, etc.) — checked by
        # action type rather than step id to handle model renumbering.
        required_non_query_actions = {
            step_meta.get("action")
            for step_id, step_meta in planned_steps.items()
            if step_meta.get("action") not in ("query", "discover")
            and _normalize_step_key(step_id) in required_ids
        }
        executed_non_query_actions = {
            v.get("action")
            for v in outcome_by_step.values()
            if _step_counts_as_completed(v, terminal_step_ids)
            and v.get("action") not in ("query", "discover")
        }

        # Non-query steps (spatial_join, aggregate) must have run when required.
        if required_non_query_actions and not required_non_query_actions.issubset(executed_non_query_actions):
            return False

        # Query steps:
        # - Single required query with no non-query steps → lenient (intersection).
        #   A single ok query is enough for simple single-db lookups.
        # - Multiple required steps → all required query steps must have run (issubset).
        if query_required_ids:
            single_query_task = len(required_ids) == 1 and not required_non_query_actions
            if single_query_task:
                if not (query_required_ids & ok_step_ids):
                    return False
            else:
                if not query_required_ids.issubset(ok_step_ids):
                    return False

    if _answer_indicates_incomplete_task(final_answer):
        return False

    if original_ids and len(required_ids) < len(original_ids) / 2:
        if len(outcome_by_step) < len(original_ids) / 2:
            return False

    return True


def _render_final_sql(
    step_registry: dict | None,
    last_validated_sql: str | None,
    last_sql: str | None,
) -> str | None:
    rendered_steps: list[str] = []

    for _, meta in _sorted_step_entries(step_registry):
        action = meta.get("action")
        step_no = meta.get("step")
        payload = dict(meta.get("step_payload") or {})

        if action == "query" and meta.get("sql"):
            database = (meta.get("database") or "").strip()
            prefix = f"-- STEP {step_no}: query"
            if database:
                prefix += f" [{database}]"
            rendered_steps.append(f"{prefix}\n{meta.get('sql')}")
            continue

        if not action:
            continue

        compact_payload = {
            k: v for k, v in payload.items()
            if v not in (None, "", [], {})
        }

        step_lines = [f"-- STEP {step_no}: {action}"]
        if compact_payload:
            step_lines.extend(
                f"-- {k}: {json.dumps(v, ensure_ascii=False, default=str)}"
                for k, v in compact_payload.items()
            )
        rendered_steps.append("\n".join(step_lines))

    if rendered_steps:
        return "\n\n-- STEP BOUNDARY --\n\n".join(rendered_steps)

    return last_validated_sql or last_sql


def _fallback_answer_from_logs(run_logs: list | None, current_run_log: list | None = None) -> str:
    candidates: list[str] = []

    for entry in current_run_log or []:
        text = (entry.get("ai_content") or "").strip()
        if text:
            candidates.append(text)

    for run in run_logs or []:
        for entry in run.get("iterations", []) or []:
            text = (entry.get("ai_content") or "").strip()
            if text:
                candidates.append(text)

    return candidates[-1] if candidates else ""


def _plan_info_from_run_logs(run_logs: list | None, fallback: dict | None = None) -> dict:
    plan = dict(fallback or {})
    source = _active_plan_run(run_logs)
    if isinstance(source, dict) and "plan_finalized" in source:
        plan = {
            "plan_finalized": source.get("plan_finalized", False),
            "planned_steps": source.get("planned_steps", {}),
            "required_step_ids": source.get("required_step_ids", []),
        }

    return {
        "plan_finalized": bool(plan.get("plan_finalized", False)),
        "planned_steps": _normalize_step_mapping(plan.get("planned_steps", {})),
        "required_step_ids": sorted(
            _normalize_step_key(step_id)
            for step_id in set(plan.get("required_step_ids", []))
        ),
    }


def _last_executed_result_from_run_logs(run_logs: list | None) -> dict | None:
    latest: dict | None = None

    for run in run_logs or []:
        for entry in run.get("iterations", []) or []:
            for tool in entry.get("tools", []) or []:
                parsed = _parse_tool_result(tool.get("result"))
                if isinstance(parsed, dict) and parsed.get("executed", False):
                    latest = parsed

    return latest


def _last_final_sql_from_run_logs(run_logs: list | None) -> str | None:
    for run in reversed(run_logs or []):
        sql = (run.get("final_sql") or "").strip()
        if sql:
            return sql
    return None


def _run_outcome_from_logs(
    run_logs: list | None,
    *,
    final_answer: str | None = None,
    valid_sql_fallback: bool = False,
    executed_fallback: bool = False,
    task_completed_fallback: bool = False,
    plan_finalized: bool = False,
    planned_steps: dict | None = None,
    required_step_ids: set | list | None = None,
    original_required_step_ids: set | list | None = None,
    final_sql_fallback: str | None = None,
    result_data_fallback: dict | None = None,
) -> dict:
    plan = _plan_info_from_run_logs(
        run_logs,
        fallback={
            "plan_finalized": plan_finalized,
            "planned_steps": planned_steps or {},
            "required_step_ids": required_step_ids or [],
        },
    )
    answer = (final_answer or "").strip() or _fallback_answer_from_logs(run_logs)
    valid_sql = _valid_sql_from_steps(
        run_logs,
        valid_sql_fallback=valid_sql_fallback,
    )
    executed = _executed_from_steps(
        run_logs,
        executed_fallback=executed_fallback,
    )
    task_completed = _task_completed_from_steps(
        run_logs,
        task_completed_fallback=task_completed_fallback,
        final_answer=answer,
        required_step_ids=set(plan["required_step_ids"]),
        plan_finalized=plan["plan_finalized"],
        planned_steps=plan["planned_steps"],
        original_required_step_ids=original_required_step_ids,
    )

    return {
        "plan_finalized": plan["plan_finalized"],
        "planned_steps": plan["planned_steps"],
        "required_step_ids": plan["required_step_ids"],
        "final_answer": answer,
        "valid_sql": valid_sql,
        "executed": executed,
        "task_completed": task_completed,
        "result_data": result_data_fallback or _last_executed_result_from_run_logs(run_logs),
        "final_sql": final_sql_fallback or _last_final_sql_from_run_logs(run_logs),
    }







def _run_log_entry_from_state(state: "AgentState") -> dict:
    steps = state.get("step_registry", {})
    final_sql = _render_final_sql(
        steps,
        state.get("last_validated_sql"),
        state.get("last_sql"),
    )
    sql_repair_counts = _normalize_step_mapping(state.get("sql_repair_counts", {}))
    step_retry_counts = _normalize_step_mapping(state.get("step_retry_counts", {}))
    normalized_step_registry = _normalize_step_mapping(state.get("step_registry", {}))

    return {
        "attempt_number": state.get("attempt_number", 0),
        "run_index": state.get("restart_count", 0),
        "model_runtime": dict(state.get("last_model_runtime", {})),
        "plan_finalized": state.get("plan_finalized", False),
        "planned_steps": _normalize_step_mapping(state.get("planned_steps", {})),
        "required_step_ids": sorted(
            _normalize_step_key(step_id)
            for step_id in set(state.get("required_step_ids", set()))
        ),
        "retrieval": {
            "scope_active": state.get("retrieval_scope_active", False),
            "top_score": state.get("retrieval_top_score"),
            "candidates": list(state.get("retrieval_candidates", [])),
        },
        "sql_repairs_used": sum(sql_repair_counts.values()),
        "sql_repair_counts": sql_repair_counts,
        "step_retries_used": sum(step_retry_counts.values()),
        "step_retry_counts": step_retry_counts,
        "replan_count": state.get("replan_count", 0),
        "replan_trigger": state.get("replan_trigger"),
        "replan_trigger_step": (
            (state.get("replan_trigger") or {}).get("zero_rows_step")
            or state.get("last_failed_step")
        ),
        "tool_shape_error_counts": dict(state.get("tool_shape_error_counts", {})),
        "unplanned_step_block_counts": dict(state.get("unplanned_step_block_counts", {})),
        "locked_step_block_counts": dict(state.get("locked_step_block_counts", {})),
        "missing_source_step_counts": dict(state.get("missing_source_step_counts", {})),
        "iterations": list(state.get("current_run_log", [])),
        "final_sql": final_sql,
        "validation_error": state.get("last_validation_error"),
        "guard_error": state.get("last_guard_error"),
        "execution_error": state.get("last_execute_error"),
        "discovered_tables": list(state.get("discovered_tables", [])),
        "discovered_databases": list(state.get("discovered_databases", [])),
        "attempt_tokens": dict(state.get("attempt_tokens", _empty_token_dict())),
        "estimated_attempt_tokens": dict(
            state.get("estimated_attempt_tokens", _empty_token_dict())
        ),
        "step_registry": normalized_step_registry,
    }





def _log_entry_key(entry: dict) -> str:
    """Stable content-based key for deduplication."""
    return json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)


def _dedup_logs(logs: list[dict]) -> list[dict]:
    """Deduplicate run logs by full content, not just run_index."""
    seen: set[str] = set()
    out: list[dict] = []
    for log in logs:
        key = _log_entry_key(log)
        if key not in seen:
            seen.add(key)
            out.append(log)
    return out


def _build_snapshot(state: "AgentState", extra_error: str | None = None) -> dict:
    errors = list(state.get("all_errors", []))
    if extra_error:
        errors.append(extra_error)

    run_logs = list(state.get("all_run_logs", []))
    if state.get("current_run_log"):
        current = _run_log_entry_from_state(state)
        if not any(_log_entry_key(r) == _log_entry_key(current) for r in run_logs):
            run_logs.append(current)
    run_logs = _dedup_logs(run_logs)

    final_sql = _render_final_sql(
        state.get("step_registry", {}),
        state.get("last_validated_sql"),
        state.get("last_sql"),
    )
    outcome = _run_outcome_from_logs(
        run_logs,
        final_answer=state.get("answer") or _fallback_answer_from_logs(run_logs),
        valid_sql_fallback=state.get("valid_sql", False),
        executed_fallback=state.get("last_execute_ok", False),
        task_completed_fallback=state.get("task_completed", False),
        plan_finalized=state.get("plan_finalized", False),
        planned_steps=dict(state.get("planned_steps", {})),
        required_step_ids=set(state.get("required_step_ids", set())),
        original_required_step_ids=set(state.get("original_required_step_ids", set())),
        final_sql_fallback=final_sql,
        result_data_fallback=state.get("last_executed_result"),
    )

    return {
        "model_id": state.get("model_id"),
        "plan_finalized": outcome["plan_finalized"],
        "planned_steps": outcome["planned_steps"],
        "required_step_ids": outcome["required_step_ids"],
        "retrieval": {
            "scope_active": state.get("retrieval_scope_active", False),
            "top_score": state.get("retrieval_top_score"),
            "candidates": list(state.get("retrieval_candidates", [])),
        },
        "valid_sql": outcome["valid_sql"],
        "executed": outcome["executed"],
        "task_completed": outcome["task_completed"],
        "answer_based_on_result": state.get("answer_based_on_result", False),
        "iterations": state.get("total_iterations", 0),
        "restart_count": state.get("restart_count", 0),
        "sql_repairs_used": sum(
            _normalize_step_mapping(state.get("sql_repair_counts", {})).values()
        ),
        "total_tool_calls": state.get("total_tool_calls", 0),
        "tokens": dict(state.get("grand_total_tokens", _empty_token_dict())),
        "estimated_tokens": dict(
            state.get("estimated_grand_total_tokens", _empty_token_dict())
        ),
        "all_errors": _dedup(errors),
        "final_sql": outcome["final_sql"],
        "run_logs": run_logs,
        "discovered_databases": list(state.get("discovered_databases", [])),
        "discovered_tables": list(state.get("discovered_tables", [])),
        "last_validation_error": state.get("last_validation_error"),
        "last_guard_error": state.get("last_guard_error"),
        "last_execute_error": state.get("last_execute_error"),
    }






def _accumulate_retry_snapshots(base: dict | None, new: dict | None) -> dict:
    """Accumulate snapshots across separate 429-retry pipeline attempts."""
    base = dict(base or {})
    new = dict(new or {})
    merged = dict(base)

    merged["model_id"] = new.get("model_id") or base.get("model_id")
    merged["iterations"] = base.get("iterations", 0) + new.get("iterations", 0)
    merged["restart_count"] = max(base.get("restart_count", 0), new.get("restart_count", 0))
    merged["sql_repairs_used"] = base.get("sql_repairs_used", 0) + new.get("sql_repairs_used", 0)
    merged["total_tool_calls"] = base.get("total_tool_calls", 0) + new.get("total_tool_calls", 0)

    bt = dict(base.get("tokens", _empty_token_dict()))
    nt = dict(new.get("tokens", _empty_token_dict()))
    merged["tokens"] = {
        "prompt_tokens":     bt["prompt_tokens"]     + nt["prompt_tokens"],
        "completion_tokens": bt["completion_tokens"] + nt["completion_tokens"],
        "total_tokens":      bt["total_tokens"]      + nt["total_tokens"],
    }
    bet = dict(base.get("estimated_tokens", _empty_token_dict()))
    net = dict(new.get("estimated_tokens", _empty_token_dict()))
    merged["estimated_tokens"] = {
        "prompt_tokens":     bet["prompt_tokens"]     + net["prompt_tokens"],
        "completion_tokens": bet["completion_tokens"] + net["completion_tokens"],
        "total_tokens":      bet["total_tokens"]      + net["total_tokens"],
    }

    merged["all_errors"] = _dedup(
        list(base.get("all_errors", [])) + list(new.get("all_errors", []))
    )
    merged["run_logs"] = _dedup_logs(
        list(base.get("run_logs", [])) + list(new.get("run_logs", []))
    )
    merged["discovered_databases"] = _dedup(
        list(base.get("discovered_databases", [])) + list(new.get("discovered_databases", []))
    )
    merged["discovered_tables"] = _dedup(
        list(base.get("discovered_tables", [])) + list(new.get("discovered_tables", []))
    )
    base_retrieval = dict(base.get("retrieval", {}))
    new_retrieval = dict(new.get("retrieval", {}))
    merged["retrieval"] = {
        "scope_active": bool(
            new_retrieval.get("scope_active", base_retrieval.get("scope_active", False))
        ),
        "top_score": (
            new_retrieval.get("top_score")
            if new_retrieval.get("top_score") is not None
            else base_retrieval.get("top_score")
        ),
        "candidates": list(new_retrieval.get("candidates") or base_retrieval.get("candidates") or []),
    }
    merged["plan_finalized"] = bool(new.get("plan_finalized", base.get("plan_finalized", False)))
    merged["planned_steps"] = _normalize_step_mapping(
        new.get("planned_steps") or base.get("planned_steps") or {}
    )
    merged["required_step_ids"] = sorted(
        {
            _normalize_step_key(step_id)
            for step_id in list(base.get("required_step_ids", [])) + list(new.get("required_step_ids", []))
        }
    )
    merged["final_sql"] = new.get("final_sql") or base.get("final_sql")
    merged["last_validation_error"] = (
        new.get("last_validation_error") or base.get("last_validation_error")
    )
    merged["last_guard_error"] = (
        new.get("last_guard_error") or base.get("last_guard_error")
    )
    merged["last_execute_error"] = (
        new.get("last_execute_error") or base.get("last_execute_error")
    )
    return merged


def _is_rate_limit_error(e: Exception) -> bool:
    try:
        import openai
        if isinstance(e, openai.RateLimitError):
            return True
    except ImportError:
        pass
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg






def _validated_sql_key(run_id: str | None, step_number: int | None) -> str | None:
    if not run_id or step_number is None:
        return None
    return f"{run_id}:{step_number}"


def _coerce_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return {}





def _increment_step_retry(state: AgentState, step_number: int) -> None:
    """Increment per-step retry counter. No-op if step_number is None."""
    if step_number is None:
        return
    counts = dict(state.get("step_retry_counts", {}))
    step_key = _normalize_step_key(step_number)
    counts[step_key] = counts.get(step_key, 0) + 1
    state["step_retry_counts"] = counts
 
 


def _increment_sql_repair(state: AgentState, step_number: int) -> None:
    if step_number is None:
        return
    counts = dict(state.get("sql_repair_counts", {}))
    step_key = _normalize_step_key(step_number)
    counts[step_key] = counts.get(step_key, 0) + 1
    state["sql_repair_counts"] = counts


def _sql_repair_ok(state: AgentState, step_number: int, max_repairs: int) -> bool:
    if step_number is None:
        return False
    step_key = _normalize_step_key(step_number)
    return state.get("sql_repair_counts", {}).get(step_key, 0) < max_repairs





def _step_retry_ok(state: AgentState, step_number: int, max_retries: int = 3) -> bool:
    """True if this step still has retry budget remaining."""
    if step_number is None:
        return False
    step_key = _normalize_step_key(step_number)
    return state.get("step_retry_counts", {}).get(step_key, 0) < max_retries
 










def _extract_from_database_catalog(parsed: Any) -> tuple[list[str], list[str]]:
    dbs = []
    tables = []

    if not isinstance(parsed, dict):
        return dbs, tables

    for db in parsed.get("databases", []):
        db_name = db.get("database")
        if db_name:
            dbs.append(db_name)

        for schema in db.get("schemas", []):
            for table in schema.get("tables", []):
                table_name = table.get("table")
                if table_name:
                    tables.append(table_name)

    return dbs, tables
