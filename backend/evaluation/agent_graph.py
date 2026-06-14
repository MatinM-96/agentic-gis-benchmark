import json
import logging
import time

from dataclasses import asdict

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ollama import ChatOllama

from config import (
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_FOUNDRY_ENDPOINT,
    AZURE_MODEL_DEPLOYMENT,
    AZURE_OPENAI_MODEL_IDS,
    AZURE_FOUNDRY_MODEL_IDS,
    OLLAMA_MODEL_IDS,
)

from agent_models import AgentState, ModelParams, AgentRunError
from backend.evaluation.prompts.agent_helpers import (
    _add_tokens,
    _answer_indicates_incomplete_task,
    _build_snapshot,
    _dedup,
    _empty_token_dict,
    _estimate_exchange_tokens,
    _executed_from_steps,
    _extract_plan_from_text,
    _extract_tokens,
    _log_entry_key,
    _run_log_entry_from_state,
    _short,
    _task_completed_from_steps,
    _valid_sql_from_steps,
)
from agent_tools_node import _tools_node, _get_tools


logger = logging.getLogger(__name__)
_PROVIDER_RETRIES_ENABLED = False






_AGENT_GRAPH = None












def _runtime_info_from_ai(model_id: str, dep: str | None, ai) -> dict:
    response_meta = dict(getattr(ai, "response_metadata", {}) or {})
    provider = response_meta.get("model_provider")
    if not provider:
        if model_id in AZURE_OPENAI_MODEL_IDS:
            provider = "azure_openai"
        elif model_id in AZURE_FOUNDRY_MODEL_IDS:
            provider = "azure_foundry"
        elif model_id in OLLAMA_MODEL_IDS:
            provider = "ollama"
        else:
            provider = "unknown"

    runtime_model_name = (
        response_meta.get("model_name")
        or response_meta.get("model")
        or dep
        or model_id
    )
    finish_reason = response_meta.get("finish_reason") or response_meta.get("done_reason")

    return {
        "provider": provider,
        "model_id": model_id,
        "deployment": dep,
        "runtime_model_name": runtime_model_name,
        "finish_reason": finish_reason,
    }


def get_model(model_id: str, tools, params: ModelParams):
    dep = AZURE_MODEL_DEPLOYMENT.get(model_id)
    provider_max_retries = params.max_retries if _PROVIDER_RETRIES_ENABLED else 0

    model_kwargs = {}
    if params.top_p is not None:
        model_kwargs["top_p"] = params.top_p
    if params.reasoning_effort is not None:
        model_kwargs["reasoning_effort"] = params.reasoning_effort

    if model_id in AZURE_OPENAI_MODEL_IDS:
        if not dep:
            raise ValueError(f"Unknown model_id: {model_id}")
        return AzureChatOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_deployment=dep,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version="2024-12-01-preview",
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            timeout=params.timeout,
            max_retries=provider_max_retries,
            model_kwargs=model_kwargs,
        ).bind_tools(tools, tool_choice=params.tool_choice)

    if model_id in AZURE_FOUNDRY_MODEL_IDS:
        if not dep:
            raise ValueError(f"Unknown model_id: {model_id}")
        return ChatOpenAI(
            model=dep,
            api_key=AZURE_OPENAI_KEY,
            base_url=AZURE_FOUNDRY_ENDPOINT,
            temperature=params.temperature,
            timeout=params.timeout,
            max_retries=provider_max_retries,
            model_kwargs=model_kwargs,
        ).bind_tools(tools, tool_choice=params.tool_choice)

    if model_id in OLLAMA_MODEL_IDS:
        return ChatOllama(
            model=model_id,
            temperature=params.temperature,
        ).bind_tools(tools)

    raise ValueError(f"Unknown model_id: {model_id}")






async def _model_node(state: AgentState) -> AgentState:
    tools = _get_tools()
    model_id = state["model_id"]
    dep = AZURE_MODEL_DEPLOYMENT.get(model_id)
    model = get_model(model_id, tools, state["params"])

    # logger.debug(
    #     "\n"
    #     + "=" * 90 + "\n"
    #     + "MODEL CALL\n"
    #     + "=" * 90 + "\n"
    #     + "  model              : %s\n"
    #     + "  attempt            : %s\n"
    #     + "  iter               : %s\n"
    #     + "  restart            : %s\n"
    #     + "  repair             : %s\n"
    #     + "  total_model_calls  : %s\n"
    #     + "  total_tools        : %s\n"
    #     + "  msg_count          : %s\n"
    #     + "  run_tokens         : %s\n"
    #     + "  grand_tokens       : %s\n"
    #     + "  last_validation_ok : %s\n"
    #     + "  last_execute_ok    : %s\n"
    #     + "  last_sql           : %s\n"
    #     + "  last_val_error     : %s\n"
    #     + "  last_exec_error    : %s\n"
    #     + "  discovered_dbs     : %s\n"
    #     + "  discovered_tables  : %s\n"
    #     + "=" * 90,
    #     state["model_id"],
    #     state.get("attempt_number", 0),
    #     state.get("iterations", 0),
    #     state.get("restart_count", 0),
    #     state.get("sql_repair_count", 0),
    #     state.get("total_iterations", 0),
    #     state.get("total_tool_calls", 0),
    #     len(state.get("msgs", [])),
    #     state.get("attempt_tokens", {}).get("total_tokens", 0),
    #     state.get("grand_total_tokens", {}).get("total_tokens", 0),
    #     state.get("last_validation_ok", False),
    #     state.get("last_execute_ok", False),
    #     _short(state.get("last_sql") or "", 120),
    #     _short(state.get("last_validation_error") or "", 120),
    #     _short(state.get("last_execute_error") or "", 120),
    #     state.get("discovered_databases", []),
    #     state.get("discovered_tables", []),
    # )
    state["crash_snapshot"] = _build_snapshot(state)

    llm_t0 = time.perf_counter()
    try:
        ai = await model.ainvoke(state["msgs"])
    except Exception as e:
        state["all_errors"] = _dedup(list(state.get("all_errors", [])) + [str(e)])
        state["crash_snapshot"] = _build_snapshot(state, extra_error=str(e))
        raise AgentRunError(str(e), snapshot=state["crash_snapshot"])
    model_latency_s = round(time.perf_counter() - llm_t0, 6)

    tool_names = [tc["name"] for tc in (ai.tool_calls or [])]
    # logger.debug(
    #     "\n"
    #     + "-" * 90 + "\n"
    #     + "MODEL RESULT\n"
    #     + "-" * 90 + "\n"
    #     + "  model              : %s\n"
    #     + "  attempt            : %s\n"
    #     + "  iteration          : %s\n"
    #     + "  tool_calls         : %s\n"
    #     + "  tool_names         : %s\n"
    #     + "  ai_content         : %s\n"
    #     + "-" * 90,
    #     state["model_id"],
    #     state.get("attempt_number", 0),
    #     state.get("iterations", 0) + 1,
    #     len(ai.tool_calls or []),
    #     tool_names if tool_names else "none",
    #     _short((ai.content or "").strip(), 200) if not ai.tool_calls else "(tool call response)",
    # )

    iter_tokens = _extract_tokens(ai)
    estimated_iter_tokens = _estimate_exchange_tokens(state["msgs"], ai)
    attempt_tokens = dict(state["attempt_tokens"])
    grand_total = dict(state["grand_total_tokens"])
    estimated_attempt_tokens = dict(
        state.get("estimated_attempt_tokens", _empty_token_dict())
    )
    estimated_grand_total = dict(
        state.get("estimated_grand_total_tokens", _empty_token_dict())
    )
    _add_tokens(attempt_tokens, iter_tokens)
    _add_tokens(grand_total, iter_tokens)
    _add_tokens(estimated_attempt_tokens, estimated_iter_tokens)
    _add_tokens(estimated_grand_total, estimated_iter_tokens)
    runtime_info = _runtime_info_from_ai(model_id, dep, ai)

    msgs = list(state["msgs"])
    msgs.append(ai)

    state["iterations"] += 1
    state["msgs"] = msgs
    state["attempt_tokens"] = attempt_tokens
    state["grand_total_tokens"] = grand_total
    state["estimated_attempt_tokens"] = estimated_attempt_tokens
    state["estimated_grand_total_tokens"] = estimated_grand_total
    state["current_ai"] = ai
    state["current_iter_tokens"] = asdict(iter_tokens)
    state["estimated_current_iter_tokens"] = asdict(estimated_iter_tokens)
    state["last_model_runtime"] = runtime_info
    state["total_iterations"] = state.get("total_iterations", 0) + 1
    state["current_ai_is_final_answer"] = not bool(ai.tool_calls)

    if not state.get("plan_finalized", False):
        planned_steps, required_step_ids = _extract_plan_from_text((ai.content or "").strip())
        if planned_steps:
            max_plan_steps = getattr(state.get("params"), "max_plan_steps", 2)
            if len(planned_steps) > max_plan_steps:
                logger.warning(
                    "      [plan] rejected — %d steps exceeds max_plan_steps=%d",
                    len(planned_steps), max_plan_steps,
                )
                state["all_errors"] = _dedup(
                    list(state.get("all_errors", []))
                    + [f"Plan rejected: {len(planned_steps)} steps exceeds limit of {max_plan_steps}. Produce a more concise plan."]
                )
                state["last_guard_error"] = json.dumps({
                    "blocked": True,
                    "reason": f"Plan has {len(planned_steps)} steps but max_plan_steps={max_plan_steps}. Simplify the plan."
                })
            else:
                # After a replan, reject a plan that only reuses already-locked steps —
                # the model would be declared complete without doing any new work.
                locked = {str(s) for s in state.get("locked_step_ids", set())}
                all_locked = (
                    state.get("replan_count", 0) > 0
                    and bool(required_step_ids)
                    and all(str(sid) in locked for sid in required_step_ids)
                )
                if all_locked:
                    logger.warning(
                        "      [plan] rejected — all required steps already locked: %s",
                        sorted(str(s) for s in required_step_ids),
                    )
                    state["all_errors"] = _dedup(
                        list(state.get("all_errors", []))
                        + ["Plan rejected: all required steps are already completed. Produce a new plan with at least one new step."]
                    )
                    state["last_guard_error"] = json.dumps({
                        "blocked": True,
                        "reason": "All steps in your plan are already executed and locked. Produce a new plan that includes at least one new step to make progress."
                    })
                else:
                    state["planned_steps"] = planned_steps
                    state["required_step_ids"] = required_step_ids
                    state["plan_finalized"] = True
                    state["plan_block_count"] = 0
                    state["current_ai_is_final_answer"] = False
                    if not state.get("original_planned_steps"):
                        state["original_planned_steps"] = dict(planned_steps)
                        state["original_required_step_ids"] = set(required_step_ids)
                    logger.info(
                        "      [plan] finalized required_step_ids=%s planned_steps=%s",
                        sorted(str(step_id) for step_id in required_step_ids),
                        sorted(str(step_id) for step_id in planned_steps.keys()),
                    )

    run_log = list(state.get("current_run_log", []))
    run_log.append({
        "iteration": state["iterations"],
        "tokens": dict(state["current_iter_tokens"]),
        "estimated_tokens": dict(state["estimated_current_iter_tokens"]),
        "model_latency_s": model_latency_s,
        "tool_latency_s": 0.0,
        "iteration_latency_s": model_latency_s,
        "ai_content": (ai.content or "").strip(),
        "llm": runtime_info,



        "tool_calls": [
            {"name": tc.get("name"), "args": tc.get("args"), "type": tc.get("type")}
            for tc in (ai.tool_calls or [])
        ],
        "tools": [],
    })
    state["current_run_log"] = run_log
    state["crash_snapshot"] = _build_snapshot(state)
    return state



















def _finalize_node(state: AgentState) -> AgentState:
    current_answer = ""
    ai = state.get("current_ai")
    if ai and state.get("current_ai_is_final_answer", False):
        candidate = (ai.content or "").strip()
        try:
            parsed_candidate = json.loads(candidate)
            if isinstance(parsed_candidate, dict) and "planned_steps" in parsed_candidate:
                candidate = ""
        except Exception:
            pass
        current_answer = candidate

    if state.get("current_run_log"):
        all_run_logs = list(state.get("all_run_logs", []))
        new_entry = _run_log_entry_from_state(state)
        new_key = _log_entry_key(new_entry)
        if not any(_log_entry_key(r) == new_key for r in all_run_logs):
            all_run_logs.append(new_entry)
        state["all_run_logs"] = all_run_logs

    run_logs = list(state.get("all_run_logs", []))
    steps = dict(state.get("step_registry", {}))
    query_steps = [s for s in steps.values() if s.get("action") == "query"]
    all_steps = list(steps.values())

    any_executed = any(s.get("executed", False) for s in all_steps)
    all_validated = (
        all(s.get("validated", False) for s in query_steps)
        if query_steps else state.get("last_validation_ok", False)
    )
    all_executed = (
        all(s.get("executed", False) for s in all_steps)
        if all_steps else state.get("last_execute_ok", False)
    )
    has_result = state.get("last_executed_result") is not None

    has_answer = bool(current_answer)
    answer_complete = not _answer_indicates_incomplete_task(current_answer)

    step_valid_sql = _valid_sql_from_steps(run_logs, valid_sql_fallback=all_validated)
    step_executed = _executed_from_steps(run_logs, executed_fallback=any_executed or all_executed)
    step_task_completed = _task_completed_from_steps(
        run_logs,
        task_completed_fallback=all_executed,
        final_answer=current_answer,
        required_step_ids=set(state.get("required_step_ids", set())),
        plan_finalized=state.get("plan_finalized", False),
        planned_steps=dict(state.get("planned_steps", {})),
        original_required_step_ids=set(state.get("original_required_step_ids", set())),
    )

    max_iter = state.get("max_iterations", 0)
    if state.get("total_iterations", 0) >= max_iter and not step_task_completed:
        state["all_errors"] = _dedup(
            list(state.get("all_errors", [])) + [f"Stopped: max_iterations ({max_iter}) reached."]
        )

    max_replans = getattr(state.get("params"), "max_replans", 1)
    if state.get("replan_trigger") is not None and state.get("replan_count", 0) >= max_replans and not step_task_completed:
        state["all_errors"] = _dedup(
            list(state.get("all_errors", [])) + [f"Stopped: replan budget ({max_replans}) exhausted."]
        )

    if not step_task_completed:
        current_answer = ""
        state["all_errors"] = _dedup(
            list(state.get("all_errors", [])) + ["Not all planned steps executed successfully."]
        )

    if step_task_completed:
        state["last_validation_error"] = None
        state["last_guard_error"] = None
        state["last_execute_error"] = None

    state["valid_sql"] = step_valid_sql
    state["last_execute_ok"] = step_executed
    state["answer"] = current_answer
    state["answer_based_on_result"] = step_task_completed and has_result and has_answer and answer_complete
    state["task_completed"] = step_task_completed
    state["done"] = True
    state["crash_snapshot"] = _build_snapshot(state)
    return state




def _route_after_model(state: AgentState) -> str:
    ai = state["current_ai"]

    if not ai.tool_calls:
        return "finalize"

    return "tools"



def _replan_node(state: AgentState) -> AgentState:
    trigger = state.get("replan_trigger") or {}
    replan_reason = trigger.get("reason")
    replan_error = trigger.get("error")
    zero_rows_step = trigger.get("zero_rows_step")
    blocked_step = state.get("last_failed_step")
    step_registry = dict(state.get("step_registry", {}))

    def _step_line(k, v):
        db = v.get("database") or ""
        action = v.get("action") or "?"
        label = f"{action}/{db}" if db else action
        rows = v.get("rows_returned")
        sql = (v.get("sql") or "").strip()
        if v.get("executed"):
            line = f"  ok:   step {k} {label} — {rows} row{'s' if rows != 1 else ''}"
            if sql and action == "query":
                line += f"\n          SQL: {sql}"
            return line
        elif v.get("validated"):
            return f"  ok:   step {k} {label} — validated, not executed"
        else:
            return f"  failed: step {k} {label} — not executed"

    step_lines = [
        _step_line(k, v)
        for k, v in sorted(step_registry.items(), key=lambda x: int(str(x[0])) if str(x[0]).isdigit() else 0)
    ]
    steps_block = "\n".join(step_lines) if step_lines else "  (none)"

    if replan_reason == "unplanned_step":
        replan_msg = f"""Replan: incomplete plan — step {blocked_step} was needed but not in the original plan.

{steps_block}

Previous results are shown above. Produce a new complete plan with fresh step numbering starting from 1.""".strip()

    elif replan_reason == "timeout_step":
        replan_msg = f"""Replan: step {blocked_step} timed out.

{steps_block}

Reason: {replan_error or "(no reason provided)"}

Produce a new complete plan with fresh step numbering starting from 1. Replace the timed-out step with a cheaper or more targeted approach.""".strip()

    elif replan_reason == "blocked_step":
        replan_msg = f"""Step {blocked_step} was blocked — {replan_error or "(no reason provided)"}.

{steps_block}

Produce a new complete plan with fresh step numbering starting from 1. Include all required analytical steps (aggregate, sort, limit).""".strip()

    else:
        replan_msg = f"""Step {zero_rows_step} returned 0 rows.

{steps_block}

Produce a new complete plan with fresh step numbering starting from 1. Options: fix the filter, try kommunenummer instead of kommunenavn, or remove the pre-filter and rely on a spatial join instead. Include all required downstream steps (aggregate, sort, limit).""".strip()

    all_run_logs = list(state.get("all_run_logs", []))
    if state.get("current_run_log"):
        new_entry = _run_log_entry_from_state(state)
        new_key = _log_entry_key(new_entry)
        if not any(_log_entry_key(r) == new_key for r in all_run_logs):
            all_run_logs.append(new_entry)

    incomplete_plan_error = (
        f"First plan was incomplete — step {blocked_step} required but not planned."
        if replan_reason == "unplanned_step" else None
    )

    # Always reset to a fresh conversation on replan. Appending a HumanMessage
    # after prior tool output can produce invalid message ordering for some APIs.
    new_msgs = list(state.get("initial_msgs") or [
        SystemMessage(content=state["prompt"]),
        HumanMessage(content=state["user_input"]),
    ])
    new_msgs.append(HumanMessage(content=replan_msg))

    shared_reset = {
        "msgs": new_msgs,
        "plan_finalized": False,
        "plan_block_count": 0,
        "unplanned_step_block_counts": {},
        "tool_shape_error_counts": {},
        "locked_step_block_counts": {},
        "missing_source_step_counts": {},
        "planned_steps": {},
        "required_step_ids": set(),
        "step_registry": {},
        "locked_step_ids": set(),
        "validated_sqls": {},
        "step_retry_counts": {},
        "sql_repair_counts": {},
        "original_planned_steps": {},
        "original_required_step_ids": set(),
        "replan_trigger": None,
        "replan_count": state.get("replan_count", 0) + 1,
        "current_ai_is_final_answer": False,
        "iter_had_error": False,
        "iter_had_validation_error": False,
        "iter_any_exec_ok": False,
        "last_validation_ok": False,
        "last_validation_error": None,
        "last_guard_error": None,
        "last_execute_error": None,
        "last_execute_ok": False,
        "last_empty_step": None,
        "last_sql": None,
        "last_validated_sql": None,
        "current_run_log": [],
        "all_run_logs": all_run_logs,
        "all_errors": _dedup(
            list(state.get("all_errors", []))
            + ([incomplete_plan_error] if incomplete_plan_error else [])
        ),
    }

    new_state = {
        **state,
        **shared_reset,
    }

    new_state["crash_snapshot"] = _build_snapshot(new_state)
    return new_state


def _route_after_tools(state: AgentState) -> str:
    max_replans = getattr(state.get("params"), "max_replans", 1)

    if state.get("replan_trigger") is not None:
        trigger = state["replan_trigger"]
        if trigger.get("reason") == "zero_rows_step":
            sig = trigger.get("zero_rows_sig", "")
            count = state.get("zero_rows_replan_counts", {}).get(sig, 0) if sig else 0
            if count > 1:
                return "finalize"  # same query returned 0 rows twice — stop looping
        if state.get("replan_count", 0) < max_replans:
            return "replan"
        return "finalize"  # replan budget exhausted

    if state.get("iterations", 0) >= state.get("max_iterations", 0):
        return "finalize"

    return "model"












def get_agent_graph():
    global _AGENT_GRAPH
    if _AGENT_GRAPH is None:
        _AGENT_GRAPH = _build_agent_graph()
    return _AGENT_GRAPH


def _build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("model", _model_node)
    graph.add_node("tools", _tools_node)
    graph.add_node("replan", _replan_node)
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("model")

    graph.add_conditional_edges(
        "model", _route_after_model,
        {"tools": "tools", "finalize": "finalize", "model": "model"},
    )
    graph.add_conditional_edges(
        "tools", _route_after_tools,
        {"model": "model", "replan": "replan", "finalize": "finalize"},
    )
    graph.add_edge("replan", "model")
    graph.add_edge("finalize", END)
    return graph.compile()
