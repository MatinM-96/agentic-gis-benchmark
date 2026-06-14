import asyncio
import logging
from pathlib import Path
import random
import sys
import time
import uuid
from dataclasses import replace

from langchain_core.messages import SystemMessage, HumanMessage


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))






from agent_models import (
    AgentState,
    AgentResult,
    PipelineResult,
    EvalRun,
    ModelParams,
    TokenUsage,
    AgentRunError,
)
from backend.evaluation.prompts.agent_helpers import (
    _accumulate_retry_snapshots,
    _build_snapshot,
    _dedup,
    _dedup_logs,
    _empty_token_dict,
    _is_rate_limit_error,
    _render_final_sql,
    _run_outcome_from_logs,
)
from agent_graph import get_agent_graph
from reporting import extract_agent_trace, _print_summary, _print_details, store_run
from backend.logging_utils import setup_logging
from backend.mcp.client.client import mcp_client
from backend.mcp.client.tools_discovery import prefetch_databases
from backend.mcp.client.tools_steps import get_step_result
from backend.embeddings.retrieve_layers import (
    build_retrieval_context,
    retrieve_database_candidates,
    load_index,
)
from config import EMBEDDINGS_MIN_SCORE, EMBEDDINGS_TOP_K, EMBEDDINGS_TABLE_INDEX_PATH



from backend.evaluation.prompts.agent_helpers import load_prompt


_MAX_RETRY_DELAY_S = 120




logger = logging.getLogger(__name__)
_PIPELINE_429_RETRIES_ENABLED = False


async def _expand_result_for_frontend(result_data: dict | None) -> dict | None:
    if not isinstance(result_data, dict):
        return result_data

    run_id = result_data.get("run_id")
    step_no = result_data.get("step")
    if not run_id or step_no is None:
        return result_data

    try:
        full_result = await get_step_result(str(run_id), int(step_no))
    except Exception as e:
        logger.warning("Failed to fetch full step result run_id=%s step=%s: %s", run_id, step_no, e)
        return result_data

    if isinstance(full_result, dict) and "error" not in full_result:
        merged = dict(result_data)
        merged.update(full_result)
        return merged

    return result_data


def _candidate_database_order(candidates: list[dict]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in candidates:
        database = str(item.get("database") or "").strip()
        if not database or database in seen:
            continue
        seen.add(database)
        ordered.append(database)
    return ordered


async def agent(
    user_input: str,
    model_id: str,
    params: ModelParams,
    prompt: str,
    attempt_number: int = 0,
    aoi_context: str | None = None,
    history_context: str | None = None,
) -> AgentResult:
    retrieval_candidates: list[dict] = []
    retrieval_context = ""
    retrieval_scope_active = False
    retrieval_top_score: float | None = None
    seeded_databases: list[str] = []
    seeded_tables: list[str] = []

    if params.use_embeddings:
        try:
            load_index(EMBEDDINGS_TABLE_INDEX_PATH)
            retrieval_candidates = retrieve_database_candidates(
                user_input,
                k=EMBEDDINGS_TOP_K,
                index_path=EMBEDDINGS_TABLE_INDEX_PATH,
            )
            top_score = retrieval_candidates[0]["score"] if retrieval_candidates else -1.0
            retrieval_top_score = top_score if retrieval_candidates else None
            retrieval_scope_active = (
                bool(retrieval_candidates) and top_score >= EMBEDDINGS_MIN_SCORE
            )
            retrieval_context = build_retrieval_context(retrieval_candidates)
            if retrieval_context and not retrieval_scope_active:
                retrieval_context += (
                    "\nFallback note: retrieval confidence is low, so broader discovery tools remain available."
                )
            logger.info(
                "[RETRIEVAL][%s] enabled=%s candidates=%s scope_active=%s top_score=%.3f",
                model_id,
                params.use_embeddings,
                [f"{c.get('database')}={c.get('score', -1):.3f}" for c in retrieval_candidates],
                retrieval_scope_active,
                top_score,
            )
        except Exception as e:
            logger.warning("[RETRIEVAL][%s] failed: %s", model_id, e)

        preferred_databases = _candidate_database_order(retrieval_candidates)
        seeded_databases = list(preferred_databases)

        if preferred_databases:
            try:
                prefetch_result = await prefetch_databases(preferred_databases)
                logger.info(
                    "[RETRIEVAL][%s] prefetched_databases=%s",
                    model_id,
                    prefetch_result.get("opened", preferred_databases)
                    if isinstance(prefetch_result, dict)
                    else preferred_databases,
                )
            except Exception as e:
                logger.warning("[RETRIEVAL][%s] database prefetch failed: %s", model_id, e)

    msgs = [SystemMessage(content=prompt)]
    if retrieval_context:
        msgs.append(SystemMessage(content=retrieval_context))
    if history_context:
        msgs.append(SystemMessage(content=history_context))
    if aoi_context:
        msgs.append(SystemMessage(content=aoi_context))
    msgs.append(HumanMessage(content=user_input))

    init_state: AgentState = {
        "user_input": user_input,
        "model_id": model_id,
        "prompt": prompt,
        "params": params,
        "msgs": msgs,
        "initial_msgs": list(msgs),
        "iterations": 0,
        "max_iterations": params.max_iterations,
        "sql_repair_count": 0,
        "max_sql_repairs": params.max_sql_repairs,
        "restart_count": 0,
        "current_run_log": [],
        "all_run_logs": [],
        "attempt_tokens": _empty_token_dict(),
        "grand_total_tokens": _empty_token_dict(),
        "current_iter_tokens": _empty_token_dict(),
        "estimated_attempt_tokens": _empty_token_dict(),
        "estimated_grand_total_tokens": _empty_token_dict(),
        "estimated_current_iter_tokens": _empty_token_dict(),
        "last_sql": None,
        "last_validated_sql": None,
        "last_validation_ok": False,
        "last_validation_error": None,
        "last_guard_error": None,
        "last_executed_result": None,
        "last_execute_ok": False,
        "last_execute_error": None,
        "last_empty_step": None,
        "discovered_tables": seeded_tables,
        "discovered_databases": [],
        "valid_sql": False,
        "answer": "",
        "all_errors": [],
        "done": False,
        "answer_based_on_result": False,
        "task_completed": False,
        "current_ai_is_final_answer": False,
        "total_sql_repairs": 0,
        "total_iterations": 0,
        "total_tool_calls": 0,
        "crash_snapshot": {},
        "attempt_number": attempt_number,
        "run_id": uuid.uuid4().hex,
        "validated_sqls": {},
        "query_mode": None,
        "step_registry": {},
        "planned_steps": {},
        "required_step_ids": set(),
        "plan_finalized": False,
        "iter_had_error": False,
        "iter_had_validation_error": False,
        "iter_any_exec_ok": False,
        "step_retry_count": 0,
        "total_step_retries": 0,
        "step_retry_counts": {},
        "last_failed_step": None,
        "locked_step_ids": set(),
        "retrieval_candidates": retrieval_candidates,
        "retrieval_scope_active": retrieval_scope_active,
        "retrieval_top_score": retrieval_top_score,


        "sql_repair_counts": {},
        "replan_trigger": None,
        "replan_count": 0,
        "unplanned_step_block_counts": {},
        "tool_shape_error_counts": {},
        "locked_step_block_counts": {},
        "missing_source_step_counts": {},
        "original_planned_steps": {},
        "original_required_step_ids": set(),
        "zero_rows_replan_counts": {},
    }

    state = init_state
    expanded_result_data: dict | None = None
    run_outcome: dict | None = None
    try:
        state = await get_agent_graph().ainvoke(init_state)
        steps = dict(state.get("step_registry", {}))
        run_logs = state.get("all_run_logs", [])
        final_sql = _render_final_sql(
            steps,
            state.get("last_validated_sql"),
            state.get("last_sql"),
        )
        run_outcome = _run_outcome_from_logs(
            run_logs,
            task_completed_fallback=state.get("task_completed", False),
            final_answer=state.get("answer"),
            valid_sql_fallback=state.get("valid_sql", False),
            executed_fallback=any(s.get("executed", False) for s in steps.values()),
            plan_finalized=state.get("plan_finalized", False),
            planned_steps=dict(state.get("planned_steps", {})),
            required_step_ids=set(state.get("required_step_ids", set())),
            original_required_step_ids=set(state.get("original_required_step_ids", set())),
            final_sql_fallback=final_sql,
            result_data_fallback=state.get("last_executed_result"),
        )
        # Expand the last executed result BEFORE cleanup_run deletes the step data.
        # _expand_result_for_frontend fetches full rows via get_step_result, which
        # would return an error after cleanup_run runs — so this must happen here.
        if run_outcome["task_completed"]:
            expanded_result_data = await _expand_result_for_frontend(
                run_outcome["result_data"]
            )
    except Exception as e:
        if isinstance(e, AgentRunError):
            raise
        snap = _build_snapshot(state, extra_error=str(e))
        raise AgentRunError(str(e), snapshot=snap)
    finally:
        try:
            await mcp_client.call("cleanup_run", {"run_id": init_state["run_id"]})
        except Exception as cleanup_err:
            logger.warning(
                "cleanup_run failed for run_id=%s: %s",
                init_state["run_id"],
                cleanup_err,
            )

    if run_outcome is None:
        raise AgentRunError("Agent run finished without a derived outcome.", snapshot=_build_snapshot(state))

    total_tokens = TokenUsage(
        prompt_tokens=state["grand_total_tokens"]["prompt_tokens"],
        completion_tokens=state["grand_total_tokens"]["completion_tokens"],
        total_tokens=state["grand_total_tokens"]["total_tokens"],
    )
    estimated_total_tokens = TokenUsage(
        prompt_tokens=state["estimated_grand_total_tokens"]["prompt_tokens"],
        completion_tokens=state["estimated_grand_total_tokens"]["completion_tokens"],
        total_tokens=state["estimated_grand_total_tokens"]["total_tokens"],
    )
    run_logs = state.get("all_run_logs", [])

    return AgentResult(
        answer=state["answer"],
        sql=run_outcome["final_sql"],
        result_data=expanded_result_data or run_outcome["result_data"],
        executed=run_outcome["executed"],
        valid_sql=run_outcome["valid_sql"],
        sql_repairs_used=state.get("total_sql_repairs", 0),
        restarts_used=state.get("restart_count", 0),
        logs=run_logs,
        all_errors=state.get("all_errors", []),
        tokens=total_tokens,
        estimated_tokens=estimated_total_tokens,
        task_completed=run_outcome["task_completed"],
        answer_based_on_result=state.get("answer_based_on_result", False),
        iterations=state.get("total_iterations", 0),
    )








async def pipeline(
    question: str,
    model_id: str,
    params: ModelParams,
    prompt: str,
    attempt_number: int = 0,
    aoi_context: str | None = None,
    history_context: str | None = None,
) -> PipelineResult:
    t0 = time.time()

    result = await agent(
        user_input=question,
        model_id=model_id,
        params=params,
        prompt=prompt,
        attempt_number=attempt_number,
        aoi_context=aoi_context,
        history_context=history_context,
    )

    return PipelineResult(
        model_id=model_id,
        final_answer=result.answer,
        final_sql=result.sql,
        result_data=result.result_data,
        valid_sql=result.valid_sql,
        executed=result.executed,
        iterations=result.iterations,
        sql_repairs_used=result.sql_repairs_used,
        restarts_used=result.restarts_used,
        latency_s=round(time.time() - t0, 3),
        all_errors=_dedup(result.all_errors),
        params=params,
        tokens=result.tokens,
        estimated_tokens=result.estimated_tokens,
        run_logs=result.logs,
        task_completed=result.task_completed,
        answer_based_on_result=result.answer_based_on_result,
    )





async def _run_one_with_retry(
    user_input: str,
    model_id: str,
    params: ModelParams,
    prompt: str,
    sem: asyncio.Semaphore,
    retries: int = 0,
    aoi_context: str | None = None,
    history_context: str | None = None,
) -> PipelineResult | AgentRunError:
    async with sem:
        effective_retries = retries if _PIPELINE_429_RETRIES_ENABLED else 0
        delay = 30
        accumulated_snapshot: dict = {
            "model_id": model_id,
            "iterations": 0,
            "restart_count": 0,
            "sql_repairs_used": 0,
            "total_tool_calls": 0,
            "tokens": _empty_token_dict(),
            "estimated_tokens": _empty_token_dict(),
            "all_errors": [],
            "final_sql": None,
            "run_logs": [],
            "discovered_databases": [],
            "discovered_tables": [],
            "plan_finalized": False,
            "plan_block_count": 0,
            "planned_steps": {},
            "required_step_ids": [],
            "retrieval": {
                "scope_active": False,
                "top_score": None,
                "candidates": [],
            },
            "last_validation_error": None,
            "last_guard_error": None,
            "last_execute_error": None,
        }

        prior_tokens = _empty_token_dict()
        prior_estimated_tokens = _empty_token_dict()
        prior_iterations = 0
        prior_run_logs: list = []

        for attempt in range(effective_retries + 1):
            try:
                result = await pipeline(
                    model_id=model_id,
                    question=user_input,
                    params=params,
                    prompt=prompt,
                    attempt_number=attempt,
                    aoi_context=aoi_context,
                    history_context=history_context,
                )
                # Carry forward cost data from failed 429 attempts
                if attempt > 0:
                    if result.tokens:
                        result.tokens = TokenUsage(
                            prompt_tokens=result.tokens.prompt_tokens + prior_tokens["prompt_tokens"],
                            completion_tokens=result.tokens.completion_tokens + prior_tokens["completion_tokens"],
                            total_tokens=result.tokens.total_tokens + prior_tokens["total_tokens"],
                        )
                    if result.estimated_tokens:
                        result.estimated_tokens = TokenUsage(
                            prompt_tokens=result.estimated_tokens.prompt_tokens + prior_estimated_tokens["prompt_tokens"],
                            completion_tokens=result.estimated_tokens.completion_tokens + prior_estimated_tokens["completion_tokens"],
                            total_tokens=result.estimated_tokens.total_tokens + prior_estimated_tokens["total_tokens"],
                        )
                    result.iterations += prior_iterations
                    if prior_run_logs and result.run_logs is not None:
                        result.run_logs = prior_run_logs + result.run_logs
                return result

            except Exception as e:
                current_snapshot = getattr(e, "snapshot", None) or {
                    "model_id": model_id,
                    "all_errors": [str(e)],
                }
                accumulated_snapshot = _accumulate_retry_snapshots(
                    accumulated_snapshot, current_snapshot
                )

                if _is_rate_limit_error(e) and attempt < effective_retries:
                    # Save cost data from this failed attempt before retrying
                    snap_tokens = current_snapshot.get("tokens") or _empty_token_dict()
                    prior_tokens = {
                        k: prior_tokens[k] + snap_tokens.get(k, 0)
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    }
                    snap_estimated_tokens = current_snapshot.get("estimated_tokens") or _empty_token_dict()
                    prior_estimated_tokens = {
                        k: prior_estimated_tokens[k] + snap_estimated_tokens.get(k, 0)
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    }
                    prior_iterations += current_snapshot.get("iterations", 0)
                    prior_run_logs = _dedup_logs(
                        prior_run_logs + list(current_snapshot.get("run_logs", []))
                    )
                    accumulated_snapshot["all_errors"] = _dedup(
                        list(accumulated_snapshot.get("all_errors", [])) + [str(e)]
                    )

                    logger.warning(
                        "[RETRY][%s] 429 detected — sleeping %ss (attempt %s/%s)...",
                        model_id, delay, attempt + 1, effective_retries + 1,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, _MAX_RETRY_DELAY_S)
                    continue

                accumulated_snapshot["all_errors"] = _dedup(
                    list(accumulated_snapshot.get("all_errors", [])) + [str(e)]
                )
                return AgentRunError(str(e), snapshot=accumulated_snapshot)








async def run_all(
    user_input: str,
    model_ids: list[str],
    params: ModelParams,
    prompt: str,
    aoi_context: str | None = None,
    history_context: str | None = None,
) -> list:
    results = []
    sem = asyncio.Semaphore(1)

    for i, model_id in enumerate(model_ids, start=1):
        logger.info("\n" + "=" * 120)
        logger.info("RUNNING MODEL %s/%s: %s", i, len(model_ids), model_id)
        logger.info("=" * 120)

        result = await _run_one_with_retry(
            user_input=user_input,
            model_id=model_id,
            params=params,
            prompt=prompt,
            sem=sem,
            retries=params.max_retries,
            aoi_context=aoi_context,
            history_context=history_context,
        )
        results.append(result)

        logger.info("\n" + "#" * 120)
        logger.info("FINAL MODEL REPORT: %s", model_id)
        logger.info("#" * 120)

        if isinstance(result, Exception):
            logger.error("[ERROR][%s] %s", model_id, result)
        else:
            _print_details([result])

    return results


async def run_sequential_benchmark(
    queries: dict[str, str],
    model_ids: list[str],
    base_params: ModelParams,
    prompt_name: str,
    prompt: str,
    no_embed_runs: int,
    embed_runs: int,
) -> None:
    runs = [(False, no_embed_runs), (True, embed_runs)]
    runs_per_model = len(queries) * (no_embed_runs + embed_runs)
    total_runs = len(model_ids) * runs_per_model
    current_run = 0

    logger.info(
        "Each model will run %s times: %s without embeddings and %s with embeddings.",
        runs_per_model,
        len(queries) * no_embed_runs,
        len(queries) * embed_runs,
    )

    for model_id in model_ids:
        for use_embeddings, repeat_count in runs:
            for repeat_no in range(1, repeat_count + 1):
                params = replace(base_params, use_embeddings=use_embeddings)

                for q_idx, (q_id, query) in enumerate(queries.items(), start=1):
                    current_run += 1
                    logger.info("=" * 120)
                    logger.info(
                        "RUN %s/%s | MODEL=%s | EMBEDDINGS=%s | REPEAT=%s/%s | QUERY %s/%s [%s]",
                        current_run,
                        total_runs,
                        model_id,
                        use_embeddings,
                        repeat_no,
                        repeat_count,
                        q_idx,
                        len(queries),
                        q_id,
                    )
                    logger.info("%s", query)
                    logger.info("=" * 120)

                    run = EvalRun(
                        query=query,
                        model_ids=[model_id],
                        prompt_name=prompt_name,
                        params=params,
                        query_id=q_id,
                    )

                    results = await run_all(
                        user_input=query,
                        model_ids=[model_id],
                        params=params,
                        prompt=prompt,
                    )

                    await store_run(run, results)
                    _print_summary(query, results)

                    if current_run < total_runs:
                        wait = random.randint(0, 3)
                        logger.info("Waiting %ss before next query...", wait)
                        await asyncio.sleep(wait)




# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
async def main():
    setup_logging()
    logger.setLevel(logging.DEBUG)


    queries = {
            # # Level 1: single-dataset grounding and simple municipality filters.
            "B1": "Count all buildings in Kristiansand municipality.",
            "B2": "Find 10 parcels in Kristiansand municipality, sorted by teigid ascending.",
            "B3": "Find 10 buildings in Narvik municipality, sorted by bygningid ascending.",
            "B4": "Count all parcels in Stavanger municipality.",
            "B5": "Count all disputed parcels in Kristiansand municipality.",
            "B6": "Count all buildings with status FA in Kristiansand municipality.",

          #  Level 2: attribute filters, numeric filters, ordering, and direct constraints.
            "B7": "Find 10 buildings with status FA in Stavanger municipality, sorted by bygningid ascending.",
            "B8": "Find 10 parcels larger than 5000 m² in Drammen municipality, sorted by parcel area descending.",
            "B9": "Find 10 parcels in Stavanger municipality that are located in agricultural land, sorted by teigid ascending.",
           "B10": "Find 10 parcels in Kristiansand municipality that intersect 200-year storm-surge areas, sorted by teigid ascending.",
            "B11": "Find 10 parcels in Trondheim municipality that intersect quick-clay hazard areas, sorted by teigid ascending.",
           "B12": "Find 10 parcels in Drammen municipality larger than 5000 m² that intersect 200-year flood zones, sorted by parcel area descending.",

            # Level 3: two-dataset spatial predicates and simple spatial aggregation.
             "B13": "Count all buildings in Narvik municipality that intersect snow-avalanche hazard areas.",
            "B14": "Count all buildings in Drammen municipality that intersect 200-year flood zones.",
             "B15": "Find 10 parcels in Drammen municipality that intersect 200-year flood zones, sorted by teigid ascending.",
            "B16": "Count all buildings in Kristiansand municipality that intersect 200-year storm-surge areas.",
           "B17": "Count all buildings in Trondheim municipality that intersect quick-clay hazard areas.",
          "B18": "Count all 1 km grid cells in Kristiansand municipality that intersect 200-year storm-surge areas.",

            # Level 4: multi-step spatial workflows with extra filters or reporting.
            "B19": "Find 10 buildings in Drammen municipality that intersect 200-year flood zones and have status FA, sorted by bygningid ascending.",
          "B20": "Find 10 buildings in Trondheim municipality that intersect quick-clay hazard areas with high risk class, sorted by bygningid ascending.",
            "B21": "Find 10 parcels in Trondheim municipality that intersect quick-clay hazard areas with high risk class, sorted by teigid ascending.",

           "B22": "Find 10 buildings in Kristiansand municipality within 50 metres of 200-year storm-surge areas, sorted by shortest distance first, and report both municipality and county.",
            "B23": "Find 10 parcels in Kristiansand municipality that intersect both 200-year storm-surge areas and 200-year flood zones, sorted by teigid ascending.",
          "B24": "Find 10 parcels in Trondheim municipality larger than 5000 m² that intersect quick-clay hazard areas with high risk class, sorted by parcel area descending, and report both municipality and county.",

           #Level 5: advanced multi-layer, distance, ranking, and top-k workflows.
            "B25": "Find 10 buildings in Kristiansand municipality that intersect both 200-year storm-surge areas and 200-year flood zones, sorted by bygningid ascending.",
           "B26": "Find 10 buildings in Drammen municipality that intersect 200-year flood zones, sorted by distance to the nearest quick-clay hazard zone ascending.",
           "B27": "Find 10 parcels in Drammen municipality larger than 5000 m² that intersect 200-year flood zones, sorted by distance to the nearest quick-clay hazard zone ascending.",
           "B28": "Find 10 buildings in Trondheim municipality that intersect quick-clay hazard areas with high risk class and are closest to the municipality centroid, sorted by shortest distance first.",
           "B29": "List the five 1 km grid cells in Lillestrøm municipality with the highest number of parcels located in quick-clay hazard areas, sorted by parcel count descending.",
           "B30": "List the five municipalities in Finnmark county with the highest number of buildings in 200-year flood zones, sorted by count descending.",

    }

  


    model_ids = [
      # "Mistral-Large-3",
       "functiongemma",
       # "gpt-5-mini",
        # "DeepSeek-V3.2",
    ]
    no_embed_runs = 20
    embed_runs = 0


    params = ModelParams(

        temperature=0.0,
        max_tokens=2000,
        timeout=300,
        max_retries=0,
        max_iterations=50,
        max_sql_repairs=2,
        max_step_retries=2,
        max_replans=2,
        max_plan_steps=10,
        use_embeddings=True,
        tool_choice="auto",
        reasoning_effort=None,
    )

    prompt_name = "prompt_v5"
    prompt = load_prompt(prompt_name)

    try:
        await run_sequential_benchmark(
            queries=queries,
            model_ids=model_ids,
            base_params=params,
            prompt_name=prompt_name,
            prompt=prompt,
            no_embed_runs=no_embed_runs,
            embed_runs=embed_runs,
        )
    finally:
        await mcp_client.close()
        load_index.cache_clear()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass






    
