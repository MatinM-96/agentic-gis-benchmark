import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from .embeddings import embed_text


_DEFAULT_INDEX_PATH = Path(__file__).parent / "table_index.json"
_SMALL_TABLE_ALL_COLUMNS_THRESHOLD = 8
_IMPORTANT_COLUMN_TOKENS = (
    "id",
    "name",
    "navn",
    "type",
    "kategori",
    "category",
    "status",
    "date",
    "dato",
    "year",
    "kode",
    "code",
    "nr",
    "nummer",
)
_GEOMETRY_TYPE_MARKERS = {"user-defined", "geometry"}


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def _normalize_columns(item: dict[str, Any]) -> list[dict[str, str]]:
    columns = item.get("selected_columns") or item.get("columns") or []
    normalized: list[dict[str, str]] = []

    for col in columns:
        if not isinstance(col, dict):
            continue
        name = str(col.get("name") or col.get("column_name") or "").strip()
        data_type = str(col.get("type") or col.get("data_type") or "").strip()
        if not name:
            continue
        normalized.append({"name": name, "type": data_type})

    return normalized


def _infer_geometry_column(item: dict[str, Any], columns: list[dict[str, str]]) -> str | None:
    geom = item.get("geometry_column") or item.get("recommended_geometry_column")
    if isinstance(geom, str) and geom.strip():
        return geom.strip()

    for col in columns:
        data_type = (col.get("type") or "").strip().lower()
        if data_type in _GEOMETRY_TYPE_MARKERS:
            return col["name"]
        if col["name"].strip().lower() in {"geom", "geometry", "geometri"}:
            return col["name"]

    return None


def _is_informative_column(name: str, data_type: str, geometry_column: str | None) -> bool:
    lowered_name = name.lower()
    lowered_type = data_type.lower()

    if geometry_column and name == geometry_column:
        return True
    if any(token in lowered_name for token in _IMPORTANT_COLUMN_TOKENS):
        return True
    if lowered_type in _GEOMETRY_TYPE_MARKERS:
        return True

    return False


def select_catalog_columns(item: dict[str, Any]) -> list[dict[str, str]]:
    columns = _normalize_columns(item)
    if len(columns) <= _SMALL_TABLE_ALL_COLUMNS_THRESHOLD:
        return columns

    geometry_column = _infer_geometry_column(item, columns)
    selected = [
        col for col in columns
        if _is_informative_column(col["name"], col.get("type", ""), geometry_column)
    ]

    if not selected:
        selected = columns[:_SMALL_TABLE_ALL_COLUMNS_THRESHOLD]

    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for col in selected:
        key = col["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(col)

    return deduped


def build_catalog_embedding_text(item: dict[str, Any]) -> str:
    database = str(item.get("database") or "").strip()
    schema = str(item.get("schema") or "").strip()
    table = str(item.get("table") or "").strip()
    description = str(item.get("short_description") or item.get("description") or "").strip()

    selected_columns = select_catalog_columns(item)
    geometry_column = _infer_geometry_column(item, _normalize_columns(item))

    column_bits = [
        f"{col['name']} ({col.get('type', '')})".strip()
        for col in selected_columns
    ]

    parts = [
        f"database: {database}",
        f"schema: {schema}",
        f"table: {table}",
    ]
    if description:
        parts.append(f"description: {description}")
    if geometry_column:
        parts.append(f"geometry column: {geometry_column}")
    if column_bits:
        parts.append(f"important columns: {', '.join(column_bits)}")

    return ". ".join(parts).strip()


def _normalize_candidate(item: dict[str, Any], score: float) -> dict[str, Any]:
    columns = _normalize_columns(item)
    geometry_column = _infer_geometry_column(item, columns)
    selected_columns = select_catalog_columns(
        {
            **item,
            "columns": columns,
            "geometry_column": geometry_column,
        }
    )

    description = str(item.get("short_description") or item.get("description") or "").strip()

    return {
        "database": str(item.get("database") or "").strip(),
        "database_description": str(item.get("database_description") or "").strip(),
        "schema": str(item.get("schema") or "").strip(),
        "table": str(item.get("table") or "").strip(),
        "full_name": str(
            item.get("full_name")
            or ".".join(
                part for part in [
                    str(item.get("database") or "").strip(),
                    str(item.get("schema") or "").strip(),
                    str(item.get("table") or "").strip(),
                ] if part
            )
        ),
        "description": description,
        "geometry_column": geometry_column,
        "selected_columns": selected_columns,
        "score": score,
    }


def _group_candidates_by_database(
    table_candidates: list[dict[str, Any]],
    per_database_table_limit: int = 3,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for item in table_candidates:
        database = str(item.get("database") or "").strip()
        if not database:
            continue

        group = grouped.setdefault(
            database,
            {
                "database": database,
                "database_description": str(item.get("database_description") or "").strip(),
                "score": -1.0,
                "supporting_tables": [],
            },
        )

        group["score"] = max(group["score"], float(item.get("score", -1.0)))
        if not group["database_description"] and item.get("database_description"):
            group["database_description"] = str(item.get("database_description") or "").strip()

        #if len(group["supporting_tables"]) < per_database_table_limit:
         #   group["supporting_tables"].append(item)

    ordered = sorted(grouped.values(), key=lambda x: x["score"], reverse=True)
    return ordered


@lru_cache(maxsize=4)
def load_index(path: str | None = None) -> list[dict]:
    index_path = Path(path) if path else _DEFAULT_INDEX_PATH
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_catalog_candidates(
    user_text: str,
    k: int = 5,
    index_path: str | None = None,
) -> list[dict[str, Any]]:
    index = load_index(index_path)
    query_vector = embed_text(user_text)

    scored: list[dict[str, Any]] = []
    for item in index:
        embedding = item.get("embedding") or []
        score = cosine(query_vector, embedding)
        scored.append(_normalize_candidate(item, score=score))

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def retrieve_database_candidates(
    user_text: str,
    k: int = 5,
    index_path: str | None = None,
    table_pool_size: int | None = None,
    per_database_table_limit: int = 3,
) -> list[dict[str, Any]]:
    table_pool_size = table_pool_size or max(k * 4, 12)
    table_candidates = retrieve_catalog_candidates(
        user_text,
        k=table_pool_size,
        index_path=index_path,
    )
    database_candidates = _group_candidates_by_database(
        table_candidates,
        per_database_table_limit=per_database_table_limit,
    )
    return database_candidates[:k]


def format_table_context(top_tables: list[dict[str, Any]]) -> str:
    return build_retrieval_context(top_tables)


def build_retrieval_context(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return ""

    lines = [
        "Pre-retrieved database candidates:",
        "Use these databases first, then decide whether the task is single_db or multi_db.",
    ]

    for item in candidates:
        lines.append(f"- {item['database']} (score={item['score']:.3f})")

    return "\n".join(lines)


def build_filtered_catalog(
    candidates: list[dict[str, Any]],
    database_name: str | None = None,
    schema_name: str | None = None,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for db_candidate in candidates:
        db = db_candidate.get("database")
        if database_name and db != database_name:
            continue

        for item in db_candidate.get("supporting_tables", []):
            schema = item.get("schema")
            if schema_name and schema != schema_name:
                continue

            grouped.setdefault((db, schema), []).append(
                {
                    "table": item.get("table"),
                    "geometry_columns": (
                        [{"column_name": item["geometry_column"]}]
                        if item.get("geometry_column") else []
                    ),
                    "recommended_geometry_column": item.get("geometry_column"),
                    "selected_columns": item.get("selected_columns", []),
                    "description": item.get("description", ""),
                }
            )

    databases: list[dict[str, Any]] = []
    by_db: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for (db, schema), tables in grouped.items():
        by_db.setdefault(db, {})[schema] = tables

    for db, schemas in sorted(by_db.items()):
        db_description = ""
        for item in candidates:
            if item.get("database") == db and item.get("database_description"):
                db_description = str(item.get("database_description") or "")
                break
        databases.append(
            {
                "database": db,
                "description": db_description,
                "schemas": [
                    {"schema": schema, "tables": tables}
                    for schema, tables in sorted(schemas.items())
                ],
            }
        )

    return {"databases": databases}


def is_candidate_in_scope(
    candidates: list[dict[str, Any]],
    database_name: str,
    schema_name: str,
    table_name: str,
) -> bool:
    for item in candidates:
        if item.get("database") == database_name:
            return True
    return False


def list_candidate_full_names(candidates: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for item in candidates:
        for table in item.get("supporting_tables", []):
            full_name = str(table.get("full_name") or "")
            if full_name:
                names.append(full_name)
    return names


def list_candidate_table_names(candidates: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    tables: list[str] = []
    for item in candidates:
        for table_item in item.get("supporting_tables", []):
            table = str(table_item.get("table") or "").strip()
            if not table or table in seen:
                continue
            seen.add(table)
            tables.append(table)
    return tables
