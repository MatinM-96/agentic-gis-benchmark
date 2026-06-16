# Agentic GIS Benchmark

Pipeline code for evaluating agentic LLM workflows on GIS benchmark tasks.

The codebase includes:

- MCP tools for database discovery and step-based GIS operations
- Agent evaluation orchestration
- Prompt versions used by the evaluation pipeline
- Retrieval support for database-context grounding
- Benchmark ground-truth task definitions

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Create a local `.env` file with the required connection strings and API keys.
The config module reads environment variables directly, including:

- `PGCONN_*` database connection strings for the GIS datasets
- `AZURE_OPENAI_KEY` and `AZURE_OPENAI_ENDPOINT`
- `AZURE_FOUNDRY_ENDPOINT` for Foundry-hosted model deployments
- `AZURE_STORAGE_CONNECTION_STRING` if results should be written to Azure Blob Storage
- `ORS_API_KEY` if OpenRouteService-backed tools are used

## Dev Container

Open the repository in a Dev Container, or start the same stack manually:

```bash
docker compose -f docker-compose.devcontainer.yml up --build
```

The compose file starts a local PostGIS container and passes through the same
environment variable names used by `config.py`. If a `PGCONN_*` variable is not
set, it falls back to the local `db` service.

## Run

Start the MCP server:

```bash
uv run python -m backend.mcp.server.run
```

Run the evaluation pipeline:

```bash
uv run python backend/evaluation/pipeline_runner.py
```

Evaluation results are written through the reporting/storage layer using the
configured `STORAGE_VERSION` and Azure Blob Storage settings.
