FROM python:3.13-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /root
COPY pyproject.toml uv.lock ./
RUN touch README.md
RUN uv sync --frozen --no-dev --no-install-project

COPY drevalpy ./drevalpy
COPY README.md ./
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.13-slim-bookworm AS runtime

LABEL image.author.name="Judith Bernett"
LABEL image.author.email="judith.bernett@tum.de"

COPY --from=builder /root/.venv /root/.venv
ENV PATH="/root/.venv/bin:$PATH"
COPY drevalpy ./drevalpy

RUN apt-get update && apt-get install -y procps unzip && rm -rf /var/lib/apt/lists/*
