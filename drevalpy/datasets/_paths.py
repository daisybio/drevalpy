"""Central cache-directory resolution for drevalpy."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir

_ENV_VAR = "DREVALPY_CACHE_DIR"


def get_default_data_dir() -> Path:
    """Return the cache directory for built-in datasets and meta.

    Resolution order:

    1. ``DREVALPY_CACHE_DIR`` environment variable (if set and non-empty).
    2. Platform-specific user cache directory via ``platformdirs``.

    :returns: Resolved cache directory path.
    """
    env = os.environ.get(_ENV_VAR, "").strip()
    if env:
        return Path(env)
    return Path(user_cache_dir("drevalpy"))
