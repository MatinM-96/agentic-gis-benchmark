FROM mcr.microsoft.com/devcontainers/python:3.12

WORKDIR /workspaces/agentic-gis-benchmark




RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    docker.io \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y nodejs




RUN pip install -U pip uv
RUN pip install -U pip npm 
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
