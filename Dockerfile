# I followed this article's recommendations
# https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0

# The builder image, used to build the virtual environment
FROM python:3.13-bookworm AS builder

RUN pip install poetry==2.1.4

# POETRY_CACHE_DIR: When removing the cache folder, make sure this is done in the same RUN command. If it’s done in a
# separate RUN command, the cache will still be part of the previous Docker layer (the one containing poetry install )
# effectively rendering your optimization useless.

ENV POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /root

COPY pyproject.toml poetry.lock ./

# First, we install only the dependencies. This way, we can cache this layer and avoid re-installing dependencies
# every time we change our application code.
# Because poetry will complain if a README.md is not found, we create a dummy one.
RUN touch README.md

RUN poetry install --without development --no-root && rm -rf $POETRY_CACHE_DIR

# The runtime image, used to run the code
FROM python:3.13-slim-bookworm AS runtime

LABEL image.author.name="Judith Bernett"
LABEL image.author.email="judith.bernett@tum.de"

# Copy installed dependencies from the builder image
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy all relevant code

COPY drevalpy ./drevalpy
COPY README.md ./
COPY pyproject.toml ./
COPY poetry.lock ./

# Install drevalpy
RUN pip install .

# Nextflow needs the command ps to be available
RUN apt-get update && apt-get install -y procps unzip && rm -rf /var/lib/apt/lists/*
