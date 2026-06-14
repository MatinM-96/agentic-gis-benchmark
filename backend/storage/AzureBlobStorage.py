import time
import json
import hashlib
import sys
from pathlib import Path
from typing import Any

from azure.storage.blob import BlobServiceClient
from azure.core import MatchConditions
from azure.core.exceptions import ResourceModifiedError, ResourceExistsError
from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER, STORAGE_VERSION

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_blob_service: BlobServiceClient | None = None

src = STORAGE_VERSION


def _get_blob_service() -> BlobServiceClient:
    global _blob_service
    if _blob_service is None:
        if not AZURE_STORAGE_CONNECTION_STRING:
            raise RuntimeError(
                "AZURE_STORAGE_CONNECTION_STRING is not set. "
                "Cannot connect to Azure Blob Storage."
            )
        _blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    return _blob_service


def query_hash(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()[:10]


def config_hash(config: Any) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def upload(path: str, data: Any):
    blob = _get_blob_service().get_blob_client(
        container=AZURE_STORAGE_CONTAINER,
        blob=path,
    )
    blob.upload_blob(json.dumps(data, indent=2, default=str), overwrite=True)


def query_meta_exists(q_hash: str) -> bool:
    try:
        _get_blob_service().get_blob_client(
            container=AZURE_STORAGE_CONTAINER,
            blob=f"{src}/{q_hash}/query.json",
        ).get_blob_properties()
        return True
    except Exception:
        return False


def prompt_meta_exists(q_hash: str, prompt_name: str) -> bool:
    try:
        _get_blob_service().get_blob_client(
            container=AZURE_STORAGE_CONTAINER,
            blob=f"{src}/{q_hash}/{prompt_name}/prompt.txt",
        ).get_blob_properties()
        return True
    except Exception:
        return False


def _download_json(path: str) -> Any:
    try:
        data = _get_blob_service().get_blob_client(
            container=AZURE_STORAGE_CONTAINER,
            blob=path,
        ).download_blob().readall()
        return json.loads(data)
    except Exception:
        return None


def _download_json_with_etag(path: str) -> tuple[Any, str | None]:
    try:
        blob_client = _get_blob_service().get_blob_client(
            container=AZURE_STORAGE_CONTAINER,
            blob=path,
        )
        stream = blob_client.download_blob()
        etag = stream.properties.etag
        return json.loads(stream.readall()), etag
    except Exception:
        return None, None


def _upload_with_etag(path: str, data: Any, etag: str | None) -> None:
    blob_client = _get_blob_service().get_blob_client(
        container=AZURE_STORAGE_CONTAINER,
        blob=path,
    )
    content = json.dumps(data, indent=2, default=str)
    if etag is not None:
        blob_client.upload_blob(
            content,
            overwrite=True,
            etag=etag,
            match_condition=MatchConditions.IfNotModified,
        )
    else:
        try:
            blob_client.upload_blob(content, overwrite=False)
        except ResourceExistsError:
            raise ResourceModifiedError("Concurrent create detected")


def _update_queries_index(
    q_hash: str,
    query: str,
    summary: list,
    prompt_name: str,
    config_id: str,
    query_id: str | None = None,
    embedding_active: bool = False,
    max_retries: int = 5,
):
    path = f"{src}/index.json"
    new_breakdown_keys = [
        f"{s['model_id']}::emb::{prompt_name}::{config_id}" if embedding_active
        else f"{s['model_id']}::{prompt_name}::{config_id}"
        for s in summary if s.get("model_id")
    ]

    for attempt in range(max_retries):
        index, etag = _download_json_with_etag(path)
        index = index or []

        if not any(e.get("query_hash") == q_hash for e in index):
            breakdown = {k: 1 for k in new_breakdown_keys}
            index.append({
                "query_id": query_id,
                "query_hash": q_hash,
                "query": query,
                "first_run": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_runs": 1,
                "breakdown": breakdown,
            })
        else:
            for e in index:
                if e.get("query_hash") == q_hash:
                    e["total_runs"] = e.get("total_runs", 0) + 1
                    if query_id and not e.get("query_id"):
                        e["query_id"] = query_id
                    breakdown = e.get("breakdown", {})
                    for k in new_breakdown_keys:
                        breakdown[k] = breakdown.get(k, 0) + 1
                    e["breakdown"] = breakdown

        try:
            _upload_with_etag(path, index, etag)
            return
        except ResourceModifiedError:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
                continue
            raise


def _update_runs_index(
    q_hash: str,
    run_id: str,
    summary: list,
    prompt_name: str,
    config_id: str,
    embedding_active: bool,
    max_retries: int = 5,
):
    path = f"{src}/{q_hash}/{prompt_name}/runs/index.json"

    for attempt in range(max_retries):
        index, etag = _download_json_with_etag(path)
        index = index or []

        index.append({
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "models_run": [s["model_id"] for s in summary],
            "prompt_name": prompt_name,
            "config_id": config_id,
            "embedding_active": embedding_active,
        })

        try:
            _upload_with_etag(path, index, etag)
            return
        except ResourceModifiedError:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
                continue
            raise
