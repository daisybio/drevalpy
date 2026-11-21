"""Central path resolution for drevalpy (cache + config directories)."""

from __future__ import annotations

import os

from platformdirs import user_cache_dir, user_config_dir
from upath import UPath as Path

_CACHE_ENV_VAR = "DREVALPY_CACHE_DIR"
_CONFIG_ENV_VAR = "DREVALPY_CONFIG_DIR"


def get_default_data_dir() -> Path:
    """Return the cache directory for built-in datasets and meta.

    Resolution order:

    1. ``DREVALPY_CACHE_DIR`` environment variable (if set and non-empty).
    2. Platform-specific user cache directory via ``platformdirs``.

    :returns: Resolved cache directory path.
    """
    env = os.environ.get(_CACHE_ENV_VAR, "").strip()
    if env:
        return Path(env)
    return Path(user_cache_dir("drevalpy"))


def get_config_dir() -> Path:
    """Return the config directory for user settings.

    Resolution order:

    1. ``DREVALPY_CONFIG_DIR`` environment variable (if set and non-empty).
    2. Platform-specific user config directory via ``platformdirs``.

    :returns: Resolved config directory path.
    """
    env = os.environ.get(_CONFIG_ENV_VAR, "").strip()
    if env:
        return Path(env)
    return Path(user_config_dir("drevalpy"))


def resolve_h5mu_path(dataset_name: str) -> Path:
    """Return the expected .h5mu path for a given dataset name.

    :param dataset_name: Name of the dataset (e.g. "GDSC1", "CTRPv2").
    :returns: Full path to the .h5mu file within the cache directory.
    """
    return get_default_data_dir() / f"{dataset_name}.h5mu"
