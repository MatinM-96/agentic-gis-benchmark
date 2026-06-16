"""Application configuration loaded from environment variables.

Most of the codebase imports these names directly, so this module keeps the
existing constant API stable while grouping related settings together.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
EMBEDDINGS_DIR = BACKEND_DIR / "embeddings"

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw_value!r}") from exc


# Database connection strings
PGCONN_STRING = _env("PGCONN_STRING")
PGCONN_MATRIKKEL = _env("PGCONN_MATRIKKEL")
PGCONN_FLOMAKTSOMRADER = _env("PGCONN_FLOMAKTSOMRADER_GIS")
PGCONN_FLOMSONER = _env("PGCONN_FLOMSONER_GIS")
PGCONN_BRANNSTASJONER = _env("PGCONN_BRANNSTASJONER_GIS")
PGCONN_KVIKKLEIRE = _env("PGCONN_KVIKKLEIRE_GIS")
PGCONN_SNOSKREDAKTSOMHETSKART = _env("PGCONN_SNOSKREDAKTSOMHETSKART_GIS")
PGCONN_MATRIKKELENEIENDOMSKARTTEIG = _env("PGCONN_MATRIKKELENEIENDOMSKARTTEIG_GIS")
PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED = _env(
    "PGCONN_AKTSOMHETSKART_JORD_FLOMSKRED_GIS"
)
PGCONN_KOMMUNER = _env("PGCONN_KOMMUNER_GIS")
PGCONN_FYLKER_GIS = _env("PGCONN_FYLKER_GIS")
PGCONN_STATISTISKRUTENETT1KM_GIS = _env("PGCONN_STATISTISKRUTENETT1KM_GIS")
PGCONN_TRAFIKKMENGDE_GIS = _env("PGCONN_TRAFIKKMENGDE_GIS")
PGCONN_STEDSNAVN_GIS = _env("PGCONN_STEDSNAVN_GIS")
PGCONN_AREALRESSURSKART_AR50 = _env("PGCONN_AREALRESSURSKART_AR50")
PGCONN_STORMFLOHAVNIVA_GIS = _env("PGCONN_STORMFLOHAVNIVA_GIS")


# API credentials and endpoints
AZURE_OPENAI_KEY = _env("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = _env("AZURE_OPENAI_ENDPOINT")
AZURE_FOUNDRY_KEY = _env("AZURE_FOUNDRY_KEY")
AZURE_FOUNDRY_ENDPOINT = _env("AZURE_FOUNDRY_ENDPOINT")
ORS_API_KEY = _env("ORS_API_KEY", "")


# Model routing
AZURE_MODEL_DEPLOYMENT = {
    # Azure OpenAI
    "gpt-35-turbo": "gpt-35-turbo",
    "gpt-35-turbo-16k": "gpt-35-turbo-16k",
    "gpt-35-turbo-instruct": "gpt-35-turbo-instruct",
    "gpt-4": "gpt-4",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1-nano": "gpt-4.1-nano",
    # Azure Foundry - GPT
    "gpt-4-32k": "gpt-4-32k",
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4.5-preview": "gpt-4.5-preview",
    "gpt-5-mini": "gpt-5-mini",
    "o1": "o1",
    "o3": "o3",
    "o3-mini": "o3-mini",
    "o3-pro": "o3-pro",
    "o4-mini": "o4-mini",
    # Azure Foundry - other providers
    "Mistral-Large-3": "Mistral-Large-3",
    "Mistral-small-2503": "mistral-small-2503",
    "Llama-4-Maverick": "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "grok-3-mini": "grok-3-mini",
    "grok-3": "grok-3",
    "DeepSeek-V3-0324": "DeepSeek-V3-0324",
    "DeepSeek-V3.1": "DeepSeek-V3.1",
    "DeepSeek-V3.2": "DeepSeek-V3.2",
    "Cohere-command-r": "Cohere-command-r-08-2024",
    "Phi-4": "Phi-4",
}

AZURE_OPENAI_MODEL_IDS = {
    "gpt-35-turbo",
    "gpt-35-turbo-16k",
    "gpt-35-turbo-instruct",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1-nano",
}

AZURE_FOUNDRY_MODEL_IDS = set(AZURE_MODEL_DEPLOYMENT) - AZURE_OPENAI_MODEL_IDS
OLLAMA_MODEL_IDS = {"functiongemma"}

# Default model subset for evaluation runs. Use list(AZURE_MODEL_DEPLOYMENT) for
# a full sweep across configured Azure deployments.
ALL_MODELS = ["Mistral-Large-3", "gpt-5-mini"]


# Embeddings
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = _env(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "text-embedding-3-large",
)
EMBEDDINGS_TOP_K = _env_int("EMBEDDINGS_TOP_K", 5)
EMBEDDINGS_MIN_SCORE = _env_float("EMBEDDINGS_MIN_SCORE", 0.20)
EMBEDDINGS_TABLE_INDEX_PATH = _env(
    "EMBEDDINGS_TABLE_INDEX_PATH",
    str(EMBEDDINGS_DIR / "table_index.json"),
)


# Blob storage
AZURE_STORAGE_CONNECTION_STRING = _env("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = _env(
    "AZURE_STORAGE_CONTAINER",
    "llm-gis-evaluation-results",
)
STORAGE_VERSION = _env("STORAGE_VERSION", "statistical_repeated_runs")


# Metadata used by database discovery and embedding index generation.
DB_DESCRIPTIONS = {
    "matrikkelenbygning": (
        "Building and cadastral building data, including building identifiers, "
        "municipality attributes, status fields, and building representation points."
    ),
    "flomaktsomrader": (
        "Flood hazard and flood susceptibility spatial data, including hazard polygons "
        "and related flood-risk layers."
    ),
    "flomsoner": (
        "Flood zone spatial data, including flood hazard polygons and flood zone identifiers."
    ),
    "kvikkleire": "Quick clay hazard data and related landslide-risk spatial layers.",
    "snoskredaktsomhetskart": (
        "Snow avalanche susceptibility data and related hazard zones."
    ),
    "matrikkeleneiendomskartteig": (
        "Property parcel data with parcel identifiers, municipality attributes, "
        "parcel polygons, and cadastral parcel metadata."
    ),
    "aktsomhetskart_jord_flomskred": (
        "Soil and flood landslide susceptibility data and related hazard layers."
    ),
    "kommuner": (
        "Municipality boundary and municipality reference data, including official "
        "municipality names, municipality numbers, and municipality polygons."
    ),
    "fylker": (
        "County boundary and county reference data, including official county names, "
        "county numbers, and county polygons."
    ),
    "statistiskrutenett1km": (
        "Regular 1 km statistical grid for spatial aggregation and grid-based analysis "
        "independent of municipality boundaries."
    ),
    "stedsnavn": (
        "Place name data with coordinates, language form, municipality affiliation, "
        "and place name type."
    ),
    "arealressurskart_ar50": (
        "AR50 land resource data with area types, agriculture classes, and forest "
        "productivity information."
    ),
    "stormflohavniva": (
        "Storm surge sea-level data for coastal risk and inundation analysis."
    ),
}
