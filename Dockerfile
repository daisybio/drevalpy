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

# procps: Nextflow needs `ps`.
# libgomp1: LightGBM's wheel links the system libgomp.so.1 by bare soname, unlike
# the xgboost and scikit-learn wheels, which vendor an auditwheel-renamed copy with
# an RPATH. Until torch's imports were deferred out of module scope, this was
# satisfied by accident: `import drevalpy` imported torch, which ships an unrenamed
# libgomp.so.1 and loads it RTLD_GLOBAL, so LightGBM's dlopen found it already in
# the link map. A lightgbm-only run never imports torch and fails without this.
RUN apt-get update \
    && apt-get install -y --no-install-recommends procps unzip libgomp1 \
    && rm -rf /var/lib/apt/lists/*

LABEL image.author.name="Judith Bernett"
LABEL image.author.email="judith.bernett@tum.de"

COPY --link --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Import each native stack in a *separate* process. A single combined import would
# pass on an image that is missing libgomp1, because importing torch first loads an
# unrenamed libgomp.so.1 RTLD_GLOBAL and resolves LightGBM's dependency for it -
# which is exactly how the missing system library stayed hidden until a
# lightgbm-only pipeline run, where torch is never imported.
RUN for module in lightgbm xgboost sklearn torch rdkit; do \
        python -c "import ${module}" || exit 1; \
    done \
    && python -c "import drevalpy"

CMD ["/bin/bash"]
