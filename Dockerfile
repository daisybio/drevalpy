FROM python:3.13-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG TORCH_BACKEND=cpu

WORKDIR /opt/build
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock ./
RUN touch README.md
RUN uv sync --frozen --no-dev --no-install-project --extra ${TORCH_BACKEND}

COPY README.md ./
COPY drevalpy ./drevalpy
RUN uv sync --frozen --no-dev --no-editable --extra ${TORCH_BACKEND}

FROM python:3.13-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y procps unzip && rm -rf /var/lib/apt/lists/*

LABEL image.author.name="Judith Bernett"
LABEL image.author.email="judith.bernett@tum.de"

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

CMD ["/bin/bash"]
