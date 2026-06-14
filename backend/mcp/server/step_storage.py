import json
from datetime import datetime, timezone

import duckdb
import pandas as pd
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from backend.mcp.server.geometry_utils import _decode_geom
from backend.mcp.server.models import AppContext






# --------------------------------------------------
# Response helpers
# --------------------------------------------------

MODEL_SAMPLE_ROWS = 5
MODEL_MAX_COLUMNS = 20
MODEL_MAX_CELL_CHARS = 200

def _err(step: int, action: str, msg: str) -> str:
    """Consistent error response — always JSON with executed=False."""
    return json.dumps({
        "executed": False,
        "step": step,
        "action": action,
        "error": str(msg),
    })

def _truncate_cell(value):
    if isinstance(value, str) and len(value) > MODEL_MAX_CELL_CHARS:
        return value[: MODEL_MAX_CELL_CHARS - 3] + "..."
    return value

_GEOM_SUFFIXES = ("_geojson", "_wkb", "_wkt")
_GEOM_NAMES = frozenset({"geom", "geometry", "geometri", "shape", "the_geom", "wkb_geometry",
                          "geom_geojson", "geometry_geojson", "omrade", "grense"})

def _is_geom_col(key: str) -> bool:
    k = key.lower()
    return k in _GEOM_NAMES or any(k.endswith(s) for s in _GEOM_SUFFIXES)

def _sample_rows(rows: list[dict], limit: int = MODEL_SAMPLE_ROWS) -> list[dict]:
    sampled: list[dict] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        sampled.append({k: _truncate_cell(v) for k, v in row.items() if not _is_geom_col(k)})
    return sampled

def _ok(run_id: str, step: int, action: str, rows: list[dict]) -> str:
    columns = list(rows[0].keys()) if rows else []
    return json.dumps({
        "executed": True,
        "run_id": run_id,
        "step": step,
        "action": action,
        "row_count": len(rows),
        "rows_returned": len(rows),
        "columns": columns[:MODEL_MAX_COLUMNS],
        "sample_rows": _sample_rows(rows),
        "result_truncated_for_model": len(rows) > MODEL_SAMPLE_ROWS,
    }, default=str)
# --------------------------------------------------
# Run store helpers
# --------------------------------------------------

def _get_run_store(
    ctx: Context[ServerSession, AppContext],
    run_id: str,
) -> dict[int, dict]:
    store = ctx.request_context.lifespan_context.step_results
    if run_id not in store:
        store[run_id] = {}
    return store[run_id]


def _get_step_entry(
    ctx: Context[ServerSession, AppContext],
    run_id: str,
    step_no: int,
) -> dict:
    run_store = _get_run_store(ctx, run_id)
    entry = run_store.get(step_no)
    if entry is None:
        raise ValueError(f"No data for step {step_no} in run {run_id}.")
    return entry


MAX_IN_MEMORY_ROWS = 5000


def _save_rows(
    ctx: Context[ServerSession, AppContext],
    run_id: str,
    step_no: int,
    rows: list[dict],
    geom_col: str | None = None,
    crs: str | None = None,
    step_meta: dict | None = None,
) -> None:
    """
    Persist step results server-side.
    step_meta captures what the agent decided — used for evaluation provenance.
    """
    store = _get_run_store(ctx, run_id)

    meta = {
        "geom_col": geom_col,
        "crs": crs,
        "row_count": len(rows),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": step_meta or {},
    }

    if len(rows) <= MAX_IN_MEMORY_ROWS:
        store[step_no] = {"storage": "memory", "rows": rows, **meta}
        return

    table_name = f"run_{run_id}_step_{step_no}"
    df = pd.DataFrame(rows)

    # DuckDB cannot serialize shapely geometry objects — convert to WKT strings first
    for col in df.columns:
        sample = df[col].dropna()
        if not sample.empty and hasattr(sample.iloc[0], "geom_type"):
            df[col] = df[col].apply(lambda g: g.wkt if hasattr(g, "geom_type") else g)
    con = None
    try:
        con = duckdb.connect(ctx.request_context.lifespan_context.duckdb_path)
        con.register("tmp_df", df)
        con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM tmp_df')
    finally:
        if con is not None:
            con.close()

    store[step_no] = {"storage": "duckdb", "table": table_name, **meta}


def _get_rows(
    ctx: Context[ServerSession, AppContext],
    run_id: str,
    step_no: int,
) -> list[dict]:
    run_store = _get_run_store(ctx, run_id)
    entry = run_store.get(step_no)

    if entry is None:
        available = sorted(run_store.keys())
        raise ValueError(
            f"No data for step {step_no} in run {run_id}. "
            f"Available steps: {available}. "
            "Ensure the source step executed successfully before referencing it."
        )

    if not isinstance(entry, dict):
        raise ValueError(f"Invalid stored result format for step {step_no} in run {run_id}.")

    storage = entry.get("storage")

    if storage == "memory":
        rows = entry.get("rows", [])
        if not isinstance(rows, list):
            raise ValueError(f"Invalid in-memory rows format for step {step_no} in run {run_id}.")
        return rows

    if storage == "duckdb":
        table_name = entry.get("table")
        if not table_name:
            raise ValueError(f"Missing DuckDB table reference for step {step_no} in run {run_id}.")
        con = None
        try:
            con = duckdb.connect(ctx.request_context.lifespan_context.duckdb_path)
            rows = con.execute(f'SELECT * FROM "{table_name}"').df().to_dict("records")
            # Re-decode WKT geometry strings that were serialized during _save_rows
            geom_col = entry.get("geom_col")
            if geom_col:
                for row in rows:
                    v = row.get(geom_col)
                    if isinstance(v, str):
                        row[geom_col] = _decode_geom(v)
            return rows
        finally:
            if con is not None:
                con.close()

    raise ValueError(f"Unknown storage type '{storage}' for step {step_no} in run {run_id}.")
