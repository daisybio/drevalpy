Bootstrap: docker
From: python:3.13-slim-bookworm

%labels
    Author Judith Bernett
    Email judith.bernett@tum.de

%environment
    # Avoid Poetry interactivity
    export POETRY_NO_INTERACTION=1

%post
    # Update and install system tools
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        procps \
        unzip \
        build-essential \
        git \
        && rm -rf /var/lib/apt/lists/*

    # Install Poetry
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="/root/.local/bin:$PATH"

    # Set working directory
    mkdir -p /opt/drevalpy
    cd /opt/drevalpy

    # Copy project files into container
    cp -r /mnt/drevalpy/* /opt/drevalpy/

    # Install only runtime dependencies
    poetry install --without development --no-root

    # Install drevalpy package
    pip install .

    # Clean up Poetry cache and pip cache
    rm -rf /root/.cache /root/.local/share/pypoetry /opt/drevalpy/README.md

%files
    drevalpy/ /mnt/drevalpy/drevalpy/
    pyproject.toml /mnt/drevalpy/pyproject.toml
    poetry.lock /mnt/drevalpy/poetry.lock
    README.md /mnt/drevalpy/README.md

%runscript
    exec /bin/bash

