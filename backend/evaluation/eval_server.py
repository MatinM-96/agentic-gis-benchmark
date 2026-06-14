#!/usr/bin/env python3
"""
eval_server.py — live evaluation front-end.

Pages
-----
  #overview   — status matrix (queries × models)
  #matrices   — model summary, failure types, stage heatmaps, DB usage
  #model/ID   — per-model: all queries, stages, failure details
  #query/HASH — per-query: all models side-by-side, repeated-run comparison
  #run/PATH   — per-run: logs.json viewer (steps, SQL, discoveries, tool sequence)

Usage
-----
  python backend/evaluation/eval_server.py
  python backend/evaluation/eval_server.py --container llm-gis-evaluation-results --storage-version prompt_comparison_evaluation --prompt prompt_v1
  python backend/evaluation/eval_server.py --host 0.0.0.0 --port 8080 --interval 60
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_STORAGE_CONTAINER,
    STORAGE_VERSION,
)

try:
    import uvicorn
    from fastapi import FastAPI, Query as QParam
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError:
    sys.exit("pip install fastapi uvicorn")

from azure.storage.blob import BlobServiceClient, ContainerClient  # noqa: E402


# ── Blob helpers ──────────────────────────────────────────────────────────────

def _container(conn: str, name: str) -> ContainerClient:
    return BlobServiceClient.from_connection_string(conn).get_container_client(name)


def _download(container: ContainerClient, path: str) -> Any:
    return json.loads(container.get_blob_client(path).download_blob().readall())


def _try_download(container: ContainerClient, path: str) -> Any:
    try:
        return _download(container, path)
    except Exception:
        return None


# ── Cache ─────────────────────────────────────────────────────────────────────

class BlobCache:
    """
    Downloads result.json blobs incrementally — only re-fetches blobs whose
    last_modified stamp changed. A background thread refreshes on the interval.
    """

    def __init__(
        self,
        *,
        connection_string: str,
        container_name: str,
        storage_version: str,
        prompt_name: str | None,
        interval: int,
    ) -> None:
        self._conn = connection_string
        self._cname = container_name
        self._sv = storage_version
        self._prompt = prompt_name
        self._interval = interval
        self._lock = threading.Lock()
        self._seen: dict[str, Any] = {}
        self._docs: dict[str, dict] = {}
        self._payload: dict[str, Any] = {}
        self._fetched_at = ""
        self._error: str | None = None
        self._refresh()
        threading.Thread(target=self._loop, daemon=True).start()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {**self._payload, "fetched_at": self._fetched_at, "error": self._error}

    def _loop(self) -> None:
        while True:
            time.sleep(self._interval)
            self._refresh()

    def _refresh(self) -> None:
        try:
            c = _container(self._conn, self._cname)
            self._pull(c)
            payload = _build(self._sv, self._prompt, self._docs)
            with self._lock:
                self._payload = payload
                self._fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self._error = None
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                self._fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _pull(self, container: ContainerClient) -> None:
        for blob in container.list_blobs(name_starts_with=f"{self._sv}/"):
            if not blob.name.endswith("/result.json"):
                continue
            parts = blob.name.split("/")
            # {sv}/{query_hash}/{prompt}/runs/{run_id}/{model_id}/result.json
            if len(parts) < 7 or parts[3] != "runs":
                continue
            if self._prompt and parts[2] != self._prompt:
                continue
            path, lm = blob.name, blob.last_modified
            if self._seen.get(path) == lm:
                continue
            doc = _try_download(container, path)
            if isinstance(doc, dict):
                doc["_blob_lm"] = lm.isoformat() if lm else ""
                self._seen[path] = lm
                self._docs[path] = doc


# ── Data build (overview) ─────────────────────────────────────────────────────

def _build(sv: str, prompt: str | None, docs: dict[str, dict]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    qtexts: dict[str, str] = {}

    for path, doc in docs.items():
        parts = path.split("/")
        qhash    = parts[1]
        run_id   = parts[4]
        model_id = parts[5] if len(parts) > 5 else (doc.get("model_id") or "unknown")
        if doc.get("query"):
            qtexts[qhash] = doc["query"]
        groups[qhash][model_id].append(_flatten(doc, path, run_id, model_id))

    all_models: list[str] = []
    queries: list[dict] = []
    for idx, qhash in enumerate(sorted(groups, key=_nk), start=1):
        models = []
        for mid in sorted(groups[qhash], key=_nk):
            runs = sorted(groups[qhash][mid], key=lambda r: r["sort_key"])
            for rno, r in enumerate(runs, start=1):
                r["run_no"] = rno
            models.append({"model_id": mid, "runs": runs})
            if mid not in all_models:
                all_models.append(mid)
        queries.append({
            "query_hash":  qhash,
            "query_label": f"B{idx}",
            "query":       qtexts.get(qhash, ""),
            "models":      models,
        })

    return {
        "queries":         queries,
        "all_models":      sorted(all_models, key=_nk),
        "blob_count":      len(docs),
        "storage_version": sv,
        "prompt_name":     prompt or "all",
    }


def _flatten(doc: dict, path: str, run_id: str, model_id: str) -> dict[str, Any]:
    process = doc.get("process")     or {}
    trace   = doc.get("trace")       or {}
    guard   = doc.get("guard_stats") or {}
    final   = doc.get("final")       or {}
    est     = doc.get("estimated_tokens") or {}
    repairs = (guard.get("sql_repairs") or {})

    completed = _b(doc.get("task_completed"))
    failure   = doc.get("final_failure_type") or ""
    fail_step = doc.get("final_failure_step")

    disc   = bool((process.get("discovery_calls") or 0) > 0
                  or trace.get("databases_explored") or trace.get("tables_explored"))
    plan   = _b(doc.get("plan_finalized"))
    sql_ok = _b(trace.get("validated_before_exec")) or _b(doc.get("valid_sql"))
    exec_  = _b(trace.get("executed_successfully")) or _b(doc.get("executed"))

    errors  = doc.get("errors") or []
    err_msg = ""
    if not completed:
        if errors:
            err_msg = str(errors[-1])[:500]
        elif doc.get("error"):
            err_msg = str(doc["error"])[:500]

    return {
        "run_id":              run_id,
        "model_id":            model_id,
        "path":                path,
        "outcome":             _outcome(completed, failure),
        "task_completed":      completed,
        "failure_type":        failure,
        "failure_step":        str(fail_step) if fail_step is not None else "",
        "iterations":          doc.get("iterations"),
        "tool_calls":          doc.get("total_tool_calls"),
        "latency_s":           doc.get("latency_s"),
        "estimated_tokens":    est.get("total_tokens"),
        "sql_repairs":         repairs.get("total") or 0,
        "replans":             guard.get("replans") or 0,
        "stages": {
            "discovery": disc,
            "planning":  plan,
            "sql_valid": sql_ok,
            "executed":  exec_,
        },
        "final_answer":        bool(final.get("answer")),
        "error_msg":           err_msg,
        # trace details for matrices — field may be list OR dict depending on pipeline version
        "databases_explored":  _as_name_list(trace.get("databases_explored")),
        "databases_in_result": _as_name_list(trace.get("databases_in_result")),
        "tables_in_result":    _as_name_list(trace.get("tables_in_result")),
        "tool_call_counts":    trace.get("tool_call_counts") or {},
        "sort_key":            doc.get("run_timestamp") or doc.get("_blob_lm") or path,
        "run_no":              0,
    }


# ── Run detail (logs.json) ────────────────────────────────────────────────────

def _build_run_detail(result_doc: dict | None, logs: Any, path: str) -> dict[str, Any]:
    """Structured view of result.json + logs.json for the run detail page."""
    if not isinstance(result_doc, dict):
        return {"error": "result.json not found", "path": path}

    parts = path.split("/")

    # Pick the first (and usually only) log run object
    log_run: dict = {}
    if isinstance(logs, list):
        for item in logs:
            if isinstance(item, dict):
                log_run = item
                break

    planned: dict = log_run.get("planned_steps") or {}
    registry: dict = log_run.get("step_registry") or {}

    clean_planned = {
        k: {
            "action":      v.get("action"),
            "description": v.get("description") or v.get("purpose"),
            "database":    v.get("database"),
        }
        for k, v in planned.items() if isinstance(v, dict)
    }

    clean_registry = {
        k: {
            "action":       v.get("action"),
            "database":     v.get("database"),
            "status":       v.get("status"),
            "validated":    v.get("validated"),
            "executed":     v.get("executed"),
            "rows_returned": v.get("rows_returned"),
            "empty_result": v.get("empty_result"),
            "sql":          v.get("sql"),
        }
        for k, v in registry.items() if isinstance(v, dict)
    }

    # Iteration log — one entry per iteration with condensed tool info
    iter_log: list[dict] = []
    for it in (log_run.get("iterations") or []):
        if not isinstance(it, dict):
            continue
        tools_out: list[dict] = []
        for t in (it.get("tools") or []):
            if not isinstance(t, dict):
                continue
            args = t.get("args") or {}
            raw_result = t.get("result")
            try:
                result = json.loads(raw_result) if isinstance(raw_result, str) else (raw_result or {})
            except Exception:
                result = {}
            if not isinstance(result, dict):
                result = {}

            tools_out.append({
                "tool":        t.get("tool"),
                "latency_s":   t.get("latency_s"),
                "db":          args.get("databaseName"),
                "table":       args.get("tableName"),
                "step":        args.get("step_number"),
                "sql":         _trunc(args.get("sql"), 600),
                "error":       _trunc(str(result.get("error") or ""), 300),
                "reason":      _trunc(str(result.get("reason") or ""), 300),
                "action":      result.get("action"),
                "tables":      result.get("tables"),     # catalog result
                "columns":     result.get("columns"),    # schema result
                "row_count":   result.get("row_count"),
                "executed":    result.get("executed"),
            })

        iter_log.append({
            "iteration":      it.get("iteration"),
            "tool_count":     len(it.get("tool_calls") or []),
            "model_latency_s": it.get("model_latency_s"),
            "tool_latency_s":  it.get("tool_latency_s"),
            "error":           _trunc(_strip_err(it.get("iteration_error")), 400),
            "failed_step":     it.get("failed_step"),
            "tools":           tools_out,
        })

    return {
        "path":     path,
        "model_id": parts[5] if len(parts) > 5 else "",
        "query_hash": parts[1] if len(parts) > 1 else "",
        "run_id":   parts[4] if len(parts) > 4 else "",
        "has_logs": bool(log_run),
        "result": {
            "task_completed":    result_doc.get("task_completed"),
            "final_failure_type": result_doc.get("final_failure_type"),
            "final_failure_step": result_doc.get("final_failure_step"),
            "iterations":        result_doc.get("iterations"),
            "total_tool_calls":  result_doc.get("total_tool_calls"),
            "latency_s":         result_doc.get("latency_s"),
            "valid_sql":         result_doc.get("valid_sql"),
            "executed":          result_doc.get("executed"),
            "query":             result_doc.get("query"),
        },
        "planned_steps":      clean_planned,
        "step_registry":      clean_registry,
        "discovered_databases": log_run.get("discovered_databases") or [],
        "discovered_tables":    log_run.get("discovered_tables") or [],
        "final_sql":            log_run.get("final_sql"),
        "iterations":           iter_log,
    }


def _trunc(value: Any, limit: int) -> str:
    if not value:
        return ""
    s = str(value)
    return s if len(s) <= limit else s[:limit - 3] + "…"


def _strip_err(value: Any) -> str:
    if not value:
        return ""
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, dict):
            return str(parsed.get("reason") or parsed.get("error") or value)
    except Exception:
        pass
    return str(value).replace("\n", " ")


def _outcome(completed: bool, failure: str) -> str:
    if completed:
        return "OK"
    return {
        "iteration_budget": "ITER",
        "replan_exhausted":  "REPLAN",
        "step_execution":    "STEP",
        "sql_validation":    "SQL",
        "pre_plan_guard":    "PLAN",
        "tool_shape":        "SHAPE",
    }.get(failure, "FAIL")


def _as_name_list(value: Any) -> list[str]:
    """Normalise databases_explored / tables_explored — pipeline stores it as either
    a list of names or a dict of {name: count}."""
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, list):
        return value
    return []


def _b(v: Any) -> bool:
    if v is None or v is False:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes"}
    return bool(v)


def _nk(s: str) -> list:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(s))]


# ── FastAPI ───────────────────────────────────────────────────────────────────

_cache: BlobCache
_conn_str: str
_cname: str

app = FastAPI()


@app.get("/api/data")
def api_data() -> JSONResponse:
    return JSONResponse(_cache.snapshot())


@app.get("/api/run")
def api_run(path: str = QParam(...)) -> JSONResponse:
    """Download result.json + logs.json for a specific run and return structured detail."""
    container = _container(_conn_str, _cname)
    result_doc = _try_download(container, path)
    logs_path  = path[: -len("result.json")] + "logs.json"
    logs       = _try_download(container, logs_path)
    return JSONResponse(_build_run_detail(result_doc, logs, path))


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_HTML)


# ── Frontend ──────────────────────────────────────────────────────────────────

_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; background: #f4f4f5; color: #111; }

/* ── Nav ── */
#nav { position: sticky; top: 0; z-index: 100; background: #fff; border-bottom: 1px solid #d4d4d4; }
.nav-inner { max-width: 1600px; margin: 0 auto; padding: 9px 20px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.brand { font-size: 14px; font-weight: 700; }
.nav-inner a { color: #333; text-decoration: none; padding: 4px 9px; border-radius: 4px; font-size: 12px; }
.nav-inner a:hover { background: #f0f0f0; }
.nav-inner a.active { background: #111; color: #fff; }
.nav-inner select { padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; background: #fff; font-family: inherit; font-size: 12px; cursor: pointer; color: #333; }
.nav-spacer { flex: 1; }
.nav-meta { color: #999; font-size: 11px; }
#countdown { color: #2563eb; font-size: 11px; font-weight: 600; }
#error-bar { background: #fef2f2; color: #991b1b; padding: 8px 20px; border-bottom: 1px solid #fca5a5; font-size: 12px; display: none; }

/* ── Layout ── */
#main { max-width: 1600px; margin: 0 auto; padding: 22px 20px 48px; }
.page-header { margin-bottom: 20px; }
.page-header h1 { font-size: 18px; font-weight: 700; margin-bottom: 6px; }
.breadcrumb { font-size: 11px; color: #999; margin-bottom: 6px; }
.breadcrumb a { color: #2563eb; text-decoration: none; }
.breadcrumb a:hover { text-decoration: underline; }
.query-full { color: #444; margin: 8px 0 6px; max-width: 900px; line-height: 1.55; }
.page-nav { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }
.stats-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }
.stat { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px 16px; min-width: 110px; }
.stat b { display: block; font-size: 18px; }
.stat span { color: #888; font-size: 11px; }

/* ── Panel ── */
.panel { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 18px 20px; margin-bottom: 18px; }
.panel h2 { font-size: 14px; font-weight: 700; margin-bottom: 10px; }
.panel h3 { font-size: 12px; font-weight: 700; margin: 14px 0 7px; color: #555; text-transform: uppercase; letter-spacing: .04em; }
.hint { color: #999; font-size: 11px; margin-bottom: 10px; }
.scroll-x { overflow-x: auto; }

/* ── Table ── */
.matrix { border-collapse: collapse; width: 100%; font-size: 12px; }
.matrix th { background: #f4f4f5; border-bottom: 2px solid #ccc; border-right: 1px solid #e0e0e0; padding: 7px 12px; text-align: left; white-space: nowrap; font-weight: 700; }
.matrix td { border-bottom: 1px solid #ebebeb; border-right: 1px solid #ebebeb; padding: 7px 12px; vertical-align: top; }
.matrix.compact td { padding: 4px 9px; text-align: center; }
.matrix tr:hover td { background: #fafafa; }
.matrix tr.row-fail > td:first-child { border-left: 3px solid #fca5a5; }
.matrix a, .matrix th a { color: #2563eb; text-decoration: none; }
.matrix a:hover, .matrix th a:hover { text-decoration: underline; }
.empty { color: #ccc; text-align: center; }
.num { text-align: right; white-space: nowrap; }
.small { font-size: 11px; }
.muted { color: #888; }
/* blue=ok, red=fail, black/grey=default — text only, no colored boxes */
.hi { color: #2563eb; } .mid { color: #888; } .lo { color: #dc2626; }
.c-ok { color: #2563eb; font-weight: 600; } .c-fail { color: #dc2626; font-weight: 600; }

/* ── Stages (plain colored text) ── */
.srow { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.sy { color: #2563eb; font-size: 11px; font-weight: 600; }
.sn { color: #dc2626; font-size: 11px; }
.s-y { color: #2563eb; text-align: center; font-weight: 600; }
.s-n { color: #dc2626; text-align: center; }

/* ── Query cell ── */
.ql { font-weight: 700; color: #111; text-decoration: none; }
.ql:hover { text-decoration: underline; }
.qtext { color: #777; margin-top: 2px; max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Error / SQL ── */
.err-block { margin-top: 7px; padding: 6px 10px; border-left: 2px solid #dc2626; color: #dc2626; font-size: 11px; white-space: pre-wrap; word-break: break-all; max-height: 120px; overflow-y: auto; line-height: 1.4; }
.sql-block { margin-top: 6px; padding: 8px 10px; background: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 4px; font-size: 11px; white-space: pre-wrap; word-break: break-all; max-height: 200px; overflow-y: auto; line-height: 1.5; color: #111; }
.pill { font-size: 11px; color: #555; margin-right: 4px; }

/* ── Tool types ── */
.tool-disc { color: #2563eb; } .tool-exec { color: #111; }
.tool-valid { color: #888; } .tool-other { color: #888; }

/* ── Run detail / expand ── */
.iter-row { cursor: pointer; }
.iter-row:hover td { background: #fafafa; }
.iter-detail td { background: #fafafa; padding: 10px 16px !important; }
.tool-card { margin-bottom: 6px; padding: 7px 10px; border-top: 1px solid #ebebeb; }
.tool-head { display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap; margin-bottom: 3px; }
.tag { font-size: 11px; color: #888; }

/* ── Buttons ── */
.btn { display: inline-block; padding: 5px 13px; border: 1px solid #d4d4d4; border-radius: 4px; background: #fff; color: #333; text-decoration: none; font-size: 12px; font-family: inherit; }
.btn:hover { background: #f5f5f5; }
.link-sm { font-size: 11px; color: #2563eb; text-decoration: none; }
.link-sm:hover { text-decoration: underline; }
</style>
</head>
<body>
<div id="nav"></div>
<div id="error-bar"></div>
<main id="main">Loading…</main>

<script>
const INTERVAL = 30;
let countdown = INTERVAL;
let lastData = null;
let expandedIters = new Set(); // for run detail page

// ── Utils ──────────────────────────────────────────────────────────────────
function e(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function outcomeCls(o) {
  return o === 'OK' ? 'c-ok' : 'c-fail';
}
function outcomeLabel(run) {
  return run.outcome === 'STEP' && run.failure_step ? `STEP@${run.failure_step}` : run.outcome;
}
function fmt(n, suf) {
  if (n == null) return '—';
  if (suf === 's') return Number(n).toFixed(1) + 's';
  if (suf === 'k') return '~' + Math.round(n/1000) + 'k';
  return String(n);
}
function pct(n, d) { return d ? Math.round(n/d*100)+'%' : '—'; }
function latestRun(mdata) { return mdata?.runs?.length ? mdata.runs[mdata.runs.length-1] : null; }

// ── Stage renderers ───────────────────────────────────────────────────────
function stagesCompact(s) {
  return [['discovery','D'],['planning','P'],['sql_valid','S'],['executed','E']]
    .map(([k,l]) => `<span class="${s[k]?'sy':'sn'}" title="${l}">${l}</span>`).join(' ');
}
function stagesFull(s) {
  return [['discovery','Disc'],['planning','Plan'],['sql_valid','SQL'],['executed','Exec']]
    .map(([k,l]) => `<span class="${s[k]?'sy':'sn'}">${l} ${s[k]?'✓':'✗'}</span>`).join('');
}

// ── Badge helpers ─────────────────────────────────────────────────────────
function runBadge(run) {
  return `<span class="${outcomeCls(run.outcome)}">R${run.run_no} ${e(outcomeLabel(run))}</span>`;
}
function viewRunLink(run) {
  return `<a class="link-sm" href="#run/${encodeURIComponent(run.path)}">logs →</a>`;
}

// ── Tool categorisation ───────────────────────────────────────────────────
function toolCls(name) {
  if (!name) return 'tool-other';
  if (name.includes('database') || name.includes('catalog') || name.includes('schema') || name.includes('table')) return 'tool-disc';
  if (name.includes('execute') || name.includes('find') || name.includes('spatial')) return 'tool-exec';
  if (name.includes('validate')) return 'tool-valid';
  return 'tool-other';
}

// ── Navigation ────────────────────────────────────────────────────────────
function renderNav(data) {
  const hash = location.hash.slice(1) || 'overview';
  const models  = data.all_models || [];
  const queries = data.queries    || [];
  const modelSel = `<select onchange="go(this.value)" title="Jump to model">
    <option value="">Model…</option>
    ${models.map(m => `<option value="model/${e(m)}" ${hash==='model/'+m?'selected':''}>${e(m)}</option>`).join('')}
  </select>`;
  const querySel = `<select onchange="go(this.value)" title="Jump to query">
    <option value="">Query…</option>
    ${queries.map(q => `<option value="query/${e(q.query_hash)}" ${hash==='query/'+q.query_hash?'selected':''}>${e(q.query_label)} — ${e(q.query.slice(0,38))}</option>`).join('')}
  </select>`;
  const a = (page, label) =>
    `<a href="#${page}" class="${hash===page||hash.startsWith(page+'/')?'active':''}">${label}</a>`;
  document.getElementById('nav').innerHTML = `<div class="nav-inner">
    <span class="brand">Eval</span>
    ${a('overview','Overview')} ${a('matrices','Matrices')}
    ${modelSel} ${querySel}
    <span class="nav-spacer"></span>
    <span class="nav-meta">${e(data.storage_version)} · ${e(data.prompt_name)}</span>
    <span class="nav-meta">${queries.length}q · ${models.length}m · ${data.blob_count||0} blobs</span>
    <span class="nav-meta" id="m-time">${e(data.fetched_at||'')}</span>
    <span id="countdown"></span>
  </div>`;
}
function go(h) { if (h) location.hash = h; }

// ── Overview ──────────────────────────────────────────────────────────────
function renderOverview(data) {
  const models  = data.all_models || [];
  const queries = data.queries    || [];
  let totalRuns=0, compRuns=0;
  for (const q of queries) for (const m of q.models) for (const r of m.runs) { totalRuns++; if (r.task_completed) compRuns++; }

  let thead = '<tr><th>Query</th>';
  for (const mid of models) {
    const s = mid.length > 20 ? mid.slice(0,18)+'…' : mid;
    thead += `<th title="${e(mid)}"><a href="#model/${e(mid)}">${e(s)}</a></th>`;
  }
  thead += '</tr>';

  let tbody = '';
  for (const q of queries) {
    let cells = `<td>
      <a class="ql" href="#query/${e(q.query_hash)}">${e(q.query_label)}</a>
      <div class="qtext" title="${e(q.query)}">${e(q.query.length>62?q.query.slice(0,59)+'…':q.query)}</div>
    </td>`;
    for (const mid of models) {
      const mdata = q.models.find(m => m.model_id === mid);
      if (!mdata) { cells += '<td class="empty">—</td>'; continue; }
      const latest = latestRun(mdata);
      const badges = mdata.runs.map(r =>
        `<a href="#query/${e(q.query_hash)}" class="${outcomeCls(r.outcome)}">${e(outcomeLabel(r))}</a>`
      ).join('  ');
      cells += `<td>${badges}<div style="margin-top:3px">${latest?stagesCompact(latest.stages):''}</div></td>`;
    }
    tbody += `<tr>${cells}</tr>`;
  }

  return `<div class="page-header">
    <h1>Overview</h1>
    <div class="stats-row">
      <div class="stat"><b>${compRuns}/${totalRuns}</b><span>runs completed</span></div>
      <div class="stat"><b>${pct(compRuns,totalRuns)}</b><span>completion rate</span></div>
      <div class="stat"><b>${queries.length}</b><span>queries</span></div>
      <div class="stat"><b>${models.length}</b><span>models</span></div>
    </div>
  </div>
  <div class="panel">
    <p class="hint">Stage icons (latest run): D=Discovery P=Planning S=SQL E=Execution. Click a query or model name for details.</p>
    <div class="scroll-x"><table class="matrix"><thead>${thead}</thead><tbody>${tbody}</tbody></table></div>
  </div>`;
}

// ── Matrices ──────────────────────────────────────────────────────────────
function renderMatrices(data) {
  const models  = data.all_models || [];
  const queries = data.queries    || [];

  // ── Table 1: Workflow stage matrix (query × model, combined DPSE per cell) ──
  const shortM=m=>m.length>18?m.slice(0,16)+'…':m;
  const stgHead=`<tr><th>Query</th>${models.map(m=>`<th title="${e(m)}">${e(shortM(m))}</th>`).join('')}</tr>`;
  const stgRows=queries.map(q=>{
    const cells=models.map(mid=>{
      const r=latestRun(q.models.find(m=>m.model_id===mid));
      if (!r) return '<td class="empty">—</td>';
      const s=r.stages;
      return `<td>${
        [['discovery','D'],['planning','P'],['sql_valid','S'],['executed','E']]
          .map(([k,l])=>`<span class="${s[k]?'sy':'sn'}">${l}</span>`)
          .join(' ')
      }</td>`;
    }).join('');
    return `<tr><td><a href="#query/${e(q.query_hash)}">${e(q.query_label)}</a></td>${cells}</tr>`;
  }).join('');

  // ── Table 2: Failure type per query × model ───────────────────────────────
  const ftHead2=`<tr><th>Query</th>${models.map(m=>`<th title="${e(m)}">${e(shortM(m))}</th>`).join('')}</tr>`;
  const ftRows=queries.map(q=>{
    const cells=models.map(mid=>{
      const r=latestRun(q.models.find(m=>m.model_id===mid));
      if (!r) return '<td class="empty">—</td>';
      if (r.task_completed) return '<td class="c-ok small">done</td>';
      return `<td class="c-fail small">${e(r.failure_type||'unknown')}</td>`;
    }).join('');
    return `<tr><td><a href="#query/${e(q.query_hash)}">${e(q.query_label)}</a></td>${cells}</tr>`;
  }).join('');

  // ── Table 3: Resource usage per model (raw counts, no percentages) ────────
  const resRows=models.map(mid=>{
    let n=0, sumItr=0, sumLat=0, latN=0, sumTok=0, tokN=0, sumRep=0, sumRepl=0;
    for (const q of queries) {
      const r=latestRun(q.models.find(m=>m.model_id===mid));
      if (!r) continue; n++;
      if (r.iterations!=null) sumItr+=r.iterations;
      if (r.latency_s!=null) { sumLat+=r.latency_s; latN++; }
      if (r.estimated_tokens!=null) { sumTok+=r.estimated_tokens; tokN++; }
      sumRep+=r.sql_repairs||0;
      sumRepl+=r.replans||0;
    }
    const avg=(sum,cnt)=>cnt?Number(sum/cnt).toFixed(1):'—';
    return `<tr>
      <td><a href="#model/${e(mid)}">${e(mid)}</a></td>
      <td class="num">${n}</td>
      <td class="num">${avg(sumItr,n)}</td>
      <td class="num">${latN?avg(sumLat,latN)+'s':'—'}</td>
      <td class="num">${tokN?Math.round(sumTok/tokN/1000)+'k':'—'}</td>
      <td class="num">${n?Number(sumRep/n).toFixed(1):'—'}</td>
      <td class="num">${n?Number(sumRepl/n).toFixed(1):'—'}</td>
    </tr>`;
  }).join('');

  // ── Table 4: Failure type count per model (crosstab) ─────────────────────
  const ftSet=new Set(), ftMap={};
  for (const mid of models) {
    ftMap[mid]={};
    for (const q of queries) {
      const r=latestRun(q.models.find(m=>m.model_id===mid));
      if (!r) continue;
      const key=r.failure_type||(r.task_completed?'workflow_complete':'unknown');
      ftSet.add(key); ftMap[mid][key]=(ftMap[mid][key]||0)+1;
    }
  }
  const ftArr=[...ftSet].sort((a,b)=>a==='workflow_complete'?-1:b==='workflow_complete'?1:a.localeCompare(b));
  const ftHead=`<tr><th>Model</th>${ftArr.map(ft=>`<th class="small">${e(ft)}</th>`).join('')}</tr>`;
  const ftBody=models.map(mid=>{
    const cells=ftArr.map(ft=>{
      const n=ftMap[mid][ft]||0;
      return `<td class="num ${n?'':'muted'}">${n||''}</td>`;
    }).join('');
    return `<tr><td><a href="#model/${e(mid)}">${e(mid)}</a></td>${cells}</tr>`;
  }).join('');

  return `<div class="page-header">
    <h1>Matrices</h1>
    <p class="muted small" style="margin-top:4px">All tables use the latest run per query/model. No ground truth — these are observable workflow metrics only.</p>
  </div>

  <div class="panel">
    <h2>Table 1 — Workflow Stages per Query × Model</h2>
    <p class="hint">D = Discovery reached · P = Planning reached · S = SQL validated · E = Executed. Blue = observed, red = not observed.</p>
    <div class="scroll-x"><table class="matrix"><thead>${stgHead}</thead><tbody>${stgRows}</tbody></table></div>
  </div>

  <div class="panel">
    <h2>Table 2 — Workflow Outcome per Query × Model</h2>
    <p class="hint">"done" = workflow completed (final answer produced). Failure type is the recorded pipeline failure, not answer correctness.</p>
    <div class="scroll-x"><table class="matrix"><thead>${ftHead2}</thead><tbody>${ftRows}</tbody></table></div>
  </div>

  <div class="panel">
    <h2>Table 3 — Resource Usage per Model</h2>
    <p class="hint">Averages across all queries (latest run). Tokens = estimated prompt+completion tokens.</p>
    <div class="scroll-x"><table class="matrix"><thead><tr>
      <th>Model</th><th>Queries</th><th>Avg iterations</th><th>Avg latency</th><th>Avg tokens</th><th>Avg SQL repairs</th><th>Avg replans</th>
    </tr></thead><tbody>${resRows}</tbody></table></div>
  </div>

  <div class="panel">
    <h2>Table 4 — Failure Type Count per Model</h2>
    <p class="hint">Count of queries (latest run) per recorded failure type. "workflow_complete" = pipeline finished without a recorded failure.</p>
    <div class="scroll-x"><table class="matrix"><thead>${ftHead}</thead><tbody>${ftBody}</tbody></table></div>
  </div>`;
}

// ── Model page ────────────────────────────────────────────────────────────
function renderModelPage(data, modelId) {
  const queries=data.queries||[], models=data.all_models||[];
  const idx=models.indexOf(modelId);
  const prev=idx>0?models[idx-1]:null, next=idx<models.length-1?models[idx+1]:null;
  const pageNav=[
    prev?`<a class="btn" href="#model/${e(prev)}">← ${e(prev)}</a>`:'',
    next?`<a class="btn" href="#model/${e(next)}">${e(next)} →</a>`:'',
  ].filter(Boolean).join('');

  let comp=0,total=0,sumLat=0,latN=0,sumIter=0;
  const rows=queries.map(q=>{
    const mdata=q.models.find(m=>m.model_id===modelId);
    if (!mdata||!mdata.runs.length) return `<tr>
      <td><a href="#query/${e(q.query_hash)}">${e(q.query_label)}</a>
        <div class="qtext small">${e(q.query.length>55?q.query.slice(0,52)+'…':q.query)}</div>
      </td><td colspan="6" class="muted small">No runs</td></tr>`;
    total++;
    const latest=latestRun(mdata);
    if (latest.task_completed) comp++;
    if (latest.latency_s!=null) { sumLat+=latest.latency_s; latN++; }
    if (latest.iterations!=null) sumIter+=latest.iterations;
    const badges=mdata.runs.map(r=>`${runBadge(r)} ${viewRunLink(r)}`).join('  ');
    const errBlock=latest.error_msg?`<div class="err-block">${e(latest.error_msg.slice(0,300))}</div>`:'';
    return `<tr class="${latest.task_completed?'':'row-fail'}">
      <td><a href="#query/${e(q.query_hash)}">${e(q.query_label)}</a>
        <div class="qtext small" title="${e(q.query)}">${e(q.query.length>55?q.query.slice(0,52)+'…':q.query)}</div>
      </td>
      <td>${badges}</td>
      <td><div class="srow">${stagesFull(latest.stages)}</div></td>
      <td class="muted small">${e(latest.failure_type||'—')}</td>
      <td class="num small">${latest.iterations!=null?latest.iterations+' itr':'—'}</td>
      <td class="num small">${fmt(latest.latency_s,'s')}</td>
    </tr>${errBlock?`<tr class="${latest.task_completed?'':'row-fail'}"><td colspan="6">${errBlock}</td></tr>`:''}`;
  }).join('');

  return `<div class="page-header">
    <div class="breadcrumb"><a href="#overview">Overview</a> / Model</div>
    <h1>${e(modelId)}</h1>
    <div class="stats-row">
      <div class="stat"><b>${comp}/${total}</b><span>completed</span></div>
      <div class="stat"><b>${pct(comp,total)}</b><span>rate</span></div>
      <div class="stat"><b>${latN?(sumLat/latN).toFixed(1)+'s':'—'}</b><span>avg latency</span></div>
      <div class="stat"><b>${total?(sumIter/total).toFixed(1)+' itr':'—'}</b><span>avg itr</span></div>
    </div>
    <div class="page-nav">${pageNav}</div>
  </div>
  <div class="panel">
    <div class="scroll-x"><table class="matrix"><thead><tr>
      <th>Query</th><th>Runs</th><th>Stages</th><th>Failure type</th><th>Itr</th><th>Latency</th>
    </tr></thead><tbody>${rows}</tbody></table></div>
  </div>`;
}

// ── Query page ────────────────────────────────────────────────────────────
function renderQueryPage(data, queryHash) {
  const queries=data.queries||[];
  const q=queries.find(q=>q.query_hash===queryHash);
  if (!q) return `<p class="muted">Query not found: ${e(queryHash)}</p>`;
  const idx=queries.indexOf(q);
  const prev=idx>0?queries[idx-1]:null, next=idx<queries.length-1?queries[idx+1]:null;
  const pageNav=[
    prev?`<a class="btn" href="#query/${e(prev.query_hash)}">← ${e(prev.query_label)}</a>`:'',
    next?`<a class="btn" href="#query/${e(next.query_hash)}">${e(next.query_label)} →</a>`:'',
  ].filter(Boolean).join('');

  const modelRows=q.models.map(m=>{
    const latest=latestRun(m);
    if (!latest) return '';
    const badges=m.runs.map(r=>`${runBadge(r)} ${viewRunLink(r)}`).join('  ');
    const allOk=m.runs.every(r=>r.task_completed), noneOk=m.runs.every(r=>!r.task_completed);
    const note=m.runs.length>1?(allOk?'all ok':noneOk?'all failed':'mixed'):'';
    const nc=note==='all ok'?'hi':note==='all failed'?'lo':note?'mid':'';
    const errBlock=latest.error_msg?`<div class="err-block">${e(latest.error_msg.slice(0,300))}</div>`:'';
    return `<tr>
      <td><a href="#model/${e(m.model_id)}">${e(m.model_id)}</a></td>
      <td>${badges}</td>
      <td><div class="srow">${stagesFull(latest.stages)}</div></td>
      <td class="muted small">${e(latest.failure_type||'—')}</td>
      <td class="num small">${latest.iterations!=null?latest.iterations+' itr':'—'}</td>
      <td class="num small">${fmt(latest.latency_s,'s')}</td>
      <td class="${nc} small">${note}</td>
    </tr>${errBlock?`<tr><td colspan="7">${errBlock}</td></tr>`:''}`;
  }).join('');

  // Repeated runs comparison
  const repeated=q.models.filter(m=>m.runs.length>1);
  const repeatSection=repeated.length?`<div class="panel">
    <h2>Repeated Runs Comparison</h2>
    <div class="scroll-x"><table class="matrix">
      <thead><tr><th>Model</th>${Array.from({length:Math.max(...repeated.map(m=>m.runs.length))},(_,i)=>`<th>R${i+1}</th>`).join('')}</tr></thead>
      <tbody>${repeated.map(m=>`<tr>
        <td><a href="#model/${e(m.model_id)}">${e(m.model_id)}</a></td>
        ${m.runs.map(r=>`<td>
          <div>${runBadge(r)} ${viewRunLink(r)}</div>
          <div class="small muted" style="margin-top:3px">${r.iterations!=null?r.iterations+' itr':''} ${fmt(r.latency_s,'s')}</div>
          <div class="srow" style="margin-top:4px">${stagesFull(r.stages)}</div>
          ${r.error_msg?`<div class="err-block">${e(r.error_msg.slice(0,200))}</div>`:''}
        </td>`).join('')}
      </tr>`).join('')}
      </tbody>
    </table></div>
  </div>`:'';

  return `<div class="page-header">
    <div class="breadcrumb"><a href="#overview">Overview</a> / Query</div>
    <h1>${e(q.query_label)}</h1>
    <p class="query-full">${e(q.query)}</p>
    <div class="muted small" style="margin-top:4px">hash: ${e(q.query_hash)}</div>
    <div class="page-nav">${pageNav}</div>
  </div>
  <div class="panel">
    <h2>All Models</h2>
    <div class="scroll-x"><table class="matrix"><thead><tr>
      <th>Model</th><th>Runs</th><th>Stages (latest)</th><th>Failure</th><th>Itr</th><th>Latency</th><th>Note</th>
    </tr></thead><tbody>${modelRows}</tbody></table></div>
  </div>
  ${repeatSection}`;
}

// ── Run detail (logs.json viewer) ─────────────────────────────────────────
function renderRunDetail(d) {
  if (d.error) return `<p class="muted">Error: ${e(d.error)}</p>`;
  const r = d.result || {};

  const outCls = r.task_completed ? 'c-ok' : 'c-fail';
  const outLabel = r.task_completed ? 'Completed' : (r.final_failure_type || 'Failed');
  const backQ  = d.query_hash ? `<a href="#query/${e(d.query_hash)}">← Query</a> /` : '';
  const backM  = d.model_id   ? `<a href="#model/${e(d.model_id)}">← Model</a> /` : '';

  // ── Planned steps vs step registry ────────────────────────────────────
  const allSteps = new Set([
    ...Object.keys(d.planned_steps||{}),
    ...Object.keys(d.step_registry||{}),
  ]);
  const stepRows=[...allSteps].sort((a,b)=>Number(a)-Number(b)).map(k=>{
    const plan=d.planned_steps?.[k]||{};
    const reg =d.step_registry?.[k]||{};
    const stsCls=reg.status==='ok'||reg.executed?'hi':reg.status==='pending'?'mid':reg.status?'lo':'';
    const sqlBlock=reg.sql?`<div class="sql-block">${e(reg.sql)}</div>`:'';
    return `<tr>
      <td class="num">${k}</td>
      <td><span class="tag">${e(plan.action||reg.action||'—')}</span></td>
      <td class="muted small">${e(plan.description||'')}</td>
      <td><span class="pill">${e(plan.database||reg.database||'—')}</span></td>
      <td class="${stsCls} small">${e(reg.status||'—')}</td>
      <td class="num small">${reg.validated?'✓':'—'}</td>
      <td class="num small">${reg.executed?'✓':'—'}</td>
      <td class="num small">${reg.rows_returned!=null?reg.rows_returned:'—'}</td>
      ${reg.empty_result?'<td class="mid small">empty</td>':'<td></td>'}
    </tr>${sqlBlock?`<tr><td></td><td colspan="8">${sqlBlock}</td></tr>`:''}`;
  }).join('');

  // ── Discovered databases + tables ─────────────────────────────────────
  const discDbs = (d.discovered_databases||[]).map(db=>`<span class="pill">${e(db)}</span>`).join(' ');
  const discTbl = d.discovered_tables||[];
  const tblList = discTbl.length
    ? `<div style="margin-top:6px;max-height:140px;overflow-y:auto">${discTbl.map(t=>`<span class="pill">${e(t)}</span>`).join(' ')}</div>`
    : '<span class="muted">none</span>';

  // ── Final SQL ─────────────────────────────────────────────────────────
  const finalSqlBlock = d.final_sql
    ? `<div class="panel"><h2>Final SQL</h2><div class="sql-block">${e(d.final_sql)}</div></div>` : '';

  // ── Iteration log ─────────────────────────────────────────────────────
  const iters = d.iterations || [];
  const iterRows = iters.map((it,i) => {
    const id = `iter-${i}`;
    const isExp = expandedIters.has(id);
    const errCls = it.error ? 'lo' : '';
    const mainRow = `<tr class="iter-row" onclick="toggleIter('${id}')">
      <td class="num">${it.iteration??i+1}</td>
      <td class="num">${it.tool_count||0}</td>
      <td class="num small">${it.model_latency_s!=null?it.model_latency_s.toFixed(2)+'s':'—'}</td>
      <td class="num small">${it.tool_latency_s!=null?it.tool_latency_s.toFixed(2)+'s':'—'}</td>
      <td class="${errCls} small">${it.failed_step!=null?'step '+it.failed_step:''} ${e((it.error||'').slice(0,100))}</td>
      <td style="color:#aaa;font-size:11px">${isExp?'▲':'▼'}</td>
    </tr>`;

    if (!isExp) return mainRow;

    const toolCards = (it.tools||[]).map(t=>{
      const tcls = toolCls(t.tool||'');
      const tags = [
        t.db    ? `<span class="tag">${e(t.db)}</span>` : '',
        t.table ? `<span class="tag">${e(t.table)}</span>` : '',
        t.step  != null ? `<span class="tag">step ${t.step}</span>` : '',
        t.latency_s!=null ? `<span class="tag">${t.latency_s.toFixed(2)}s</span>` : '',
      ].filter(Boolean).join(' ');

      const cols = Array.isArray(t.columns)
        ? `<div style="margin-top:5px"><b>Columns:</b> ${t.columns.map(c=>`<span class="pill">${e(c)}</span>`).join(' ')}</div>` : '';
      const tbls = Array.isArray(t.tables)
        ? `<div style="margin-top:5px"><b>Tables:</b> ${t.tables.slice(0,20).map(tb=>`<span class="pill">${e(tb)}</span>`).join(' ')}${t.tables.length>20?' …':''}</div>` : '';
      const sqlCard = t.sql ? `<div class="sql-block" style="margin-top:5px">${e(t.sql)}</div>` : '';
      const errCard = (t.error||t.reason)
        ? `<div class="err-block" style="margin-top:5px">${e(t.error||t.reason)}</div>` : '';
      const execTag = t.executed!=null ? `<span class="tag ${t.executed?'hi':'lo'}">executed: ${t.executed}</span>` : '';
      const rowTag  = t.row_count!=null ? `<span class="tag">${t.row_count} rows</span>` : '';
      const actionTag = t.action ? `<span class="tag">${e(t.action)}</span>` : '';

      return `<div class="tool-card">
        <div class="tool-head">
          <span class="${tcls}" style="font-weight:700">${e(t.tool||'?')}</span>
          ${tags} ${execTag} ${rowTag} ${actionTag}
        </div>
        ${cols}${tbls}${sqlCard}${errCard}
      </div>`;
    }).join('');

    return mainRow + `<tr class="iter-detail"><td colspan="6">
      ${it.error?`<div class="err-block" style="margin-bottom:8px">${e(it.error)}</div>`:''}
      ${toolCards||'<span class="muted small">No tool details recorded.</span>'}
    </td></tr>`;
  }).join('');

  return `<div class="page-header">
    <div class="breadcrumb">${backM} ${backQ} Run log</div>
    <h1>${e(d.model_id)}</h1>
    <p class="muted small" style="margin:4px 0">${e(d.path)}</p>
    <div class="stats-row" style="margin-top:10px">
      <div class="stat"><b><span class="${outCls}">${e(outLabel)}</span></b><span>outcome</span></div>
      <div class="stat"><b>${r.iterations??'—'}</b><span>iterations</span></div>
      <div class="stat"><b>${r.total_tool_calls??'—'}</b><span>tool calls</span></div>
      <div class="stat"><b>${r.latency_s!=null?r.latency_s.toFixed(1)+'s':'—'}</b><span>latency</span></div>
      ${r.final_failure_type?`<div class="stat"><b>${e(r.final_failure_type)}</b><span>failure type</span></div>`:''}
      ${r.final_failure_step!=null?`<div class="stat"><b>step ${r.final_failure_step}</b><span>failure step</span></div>`:''}
    </div>
    ${!d.has_logs?'<p class="muted small" style="margin-top:8px">logs.json not found for this run.</p>':''}
  </div>

  <div class="panel">
    <h2>Planned Steps vs Step Registry</h2>
    <p class="hint">SQL from step_registry (validated/executed). Click a step row to see SQL inline.</p>
    <div class="scroll-x"><table class="matrix">
      <thead><tr><th>#</th><th>Action</th><th>Description</th><th>Database</th><th>Status</th><th>Valid</th><th>Exec</th><th>Rows</th><th></th></tr></thead>
      <tbody>${stepRows||'<tr><td colspan="9" class="muted">No step data in logs.</td></tr>'}</tbody>
    </table></div>
  </div>

  <div class="panel">
    <h2>Discovery</h2>
    <h3>Databases (${(d.discovered_databases||[]).length})</h3>
    <div>${discDbs||'<span class="muted">none recorded</span>'}</div>
    <h3>Tables (${discTbl.length})</h3>
    ${tblList}
  </div>

  ${finalSqlBlock}

  <div class="panel">
    <h2>Iteration Log</h2>
    <p class="hint">Click a row to expand tool calls for that iteration. D=discovery tools, E=execution, S=SQL validation.</p>
    <table class="matrix">
      <thead><tr><th>Iter</th><th>Tools</th><th>Model s</th><th>Tool s</th><th>Error / note</th><th></th></tr></thead>
      <tbody>${iterRows||'<tr><td colspan="6" class="muted">No iterations recorded.</td></tr>'}</tbody>
    </table>
  </div>`;
}

function toggleIter(id) {
  if (expandedIters.has(id)) expandedIters.delete(id);
  else expandedIters.add(id);
  // re-render current run detail without re-fetching
  if (currentRunData) document.getElementById('main').innerHTML = renderRunDetail(currentRunData);
}

let currentRunData = null;
async function loadRunDetail(encodedPath) {
  expandedIters.clear();
  currentRunData = null;
  document.getElementById('main').innerHTML = '<p style="padding:20px;color:#888">Loading run detail…</p>';
  try {
    const path = decodeURIComponent(encodedPath);
    const d = await fetch('/api/run?path=' + encodeURIComponent(path)).then(r => r.json());
    currentRunData = d;
    if (lastData) renderNav(lastData);
    document.getElementById('main').innerHTML = renderRunDetail(d);
  } catch(err) {
    document.getElementById('main').innerHTML = `<p class="muted" style="padding:20px">Error: ${e(String(err))}</p>`;
  }
}

// ── Router ────────────────────────────────────────────────────────────────
function route(data) {
  const hash = location.hash.slice(1) || 'overview';
  if (hash === 'matrices')        return renderMatrices(data);
  if (hash.startsWith('model/'))  return renderModelPage(data, decodeURIComponent(hash.slice(6)));
  if (hash.startsWith('query/'))  return renderQueryPage(data, hash.slice(6));
  if (hash.startsWith('run/'))    { loadRunDetail(hash.slice(4)); return null; }
  return renderOverview(data);
}

// ── Render ────────────────────────────────────────────────────────────────
function render(data) {
  lastData = data;
  try { renderNav(data); } catch(err) { console.error('renderNav error:', err); }
  try {
    const content = route(data);
    if (content !== null) {
      currentRunData = null;
      document.getElementById('main').innerHTML = content;
    }
  } catch(err) {
    console.error('render error:', err);
    document.getElementById('main').innerHTML =
      `<pre style="padding:20px;color:#dc2626;white-space:pre-wrap">${e(String(err))}\n${e(err.stack||'')}</pre>`;
  }
  const bar = document.getElementById('error-bar');
  if (data.error) { bar.textContent = 'Error: ' + data.error; bar.style.display = ''; }
  else bar.style.display = 'none';
}

window.onerror = (msg, src, line, col, err) => {
  console.error('window.onerror', msg, err);
};

// ── Auto-refresh ──────────────────────────────────────────────────────────
async function refresh() {
  try {
    const data = await fetch('/api/data').then(r => r.json());
    render(data);
    countdown = INTERVAL;
  } catch(err) {
    const bar = document.getElementById('error-bar');
    bar.textContent = 'Fetch error: ' + err;
    bar.style.display = '';
  }
}

setInterval(() => {
  countdown--;
  const el = document.getElementById('countdown');
  if (el) el.textContent = `refresh in ${countdown}s`;
  if (countdown <= 0) refresh();
}, 1000);

window.addEventListener('hashchange', () => {
  const hash = location.hash.slice(1);
  if (hash.startsWith('run/')) { loadRunDetail(hash.slice(4)); return; }
  if (lastData) render(lastData);
});

refresh();
</script>
</body>
</html>
"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global _cache, _conn_str, _cname

    parser = argparse.ArgumentParser(description="Live evaluation dashboard.")
    parser.add_argument("--storage-version", default=STORAGE_VERSION)
    parser.add_argument("--prompt",    default=None, help="Filter to one prompt (e.g. prompt_v1).")
    parser.add_argument("--container", default=AZURE_STORAGE_CONTAINER)
    parser.add_argument("--interval",  type=int, default=30, help="Background refresh interval (seconds).")
    parser.add_argument("--host",      default="127.0.0.1")
    parser.add_argument("--port",      type=int, default=8000)
    args = parser.parse_args()

    if not AZURE_STORAGE_CONNECTION_STRING:
        sys.exit("AZURE_STORAGE_CONNECTION_STRING is not set.")

    _conn_str = AZURE_STORAGE_CONNECTION_STRING
    _cname    = args.container

    print(f"Loading blobs — container={args.container!r}  version={args.storage_version!r} …")
    _cache = BlobCache(
        connection_string=AZURE_STORAGE_CONNECTION_STRING,
        container_name=args.container,
        storage_version=args.storage_version,
        prompt_name=args.prompt,
        interval=args.interval,
    )

    snap = _cache.snapshot()
    print(
        f"Ready — {snap.get('blob_count',0)} blobs · "
        f"{len(snap.get('queries',[]))} queries · "
        f"{len(snap.get('all_models',[]))} models"
    )
    print(f"Dashboard: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
