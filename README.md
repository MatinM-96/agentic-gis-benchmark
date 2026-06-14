# Agentic GIS Benchmark

This repository contains the pipeline code for evaluating agentic LLM workflows on GIS benchmark tasks.

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

Create a local `.env` file with the required database, Azure OpenAI, storage, and embedding configuration values.

## Run

Start the MCP server:

```bash
uv run python -m backend.mcp.server.run
```

Run the evaluation pipeline:

```bash
uv run python backend/evaluation/pipeline_runner.py
```

Start the evaluation dashboard:

```bash
uv run python backend/evaluation/eval_server.py
```
