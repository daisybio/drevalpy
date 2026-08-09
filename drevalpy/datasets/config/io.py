"""Config file I/O for drevalpy user configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path

from platformdirs import user_config_dir

from .models import DrevalConfig

_ENV_VAR = "DREVALPY_CONFIG_DIR"


def get_config_path() -> Path:
    """Return path to the user config file.

    Checks ``DREVALPY_CONFIG_DIR`` env var first, falls back to platformdirs.

    :returns: Path to ``drevalpy.json``.
    """
    env = os.environ.get(_ENV_VAR, "").strip()
    config_dir = Path(env) if env else Path(user_config_dir("drevalpy"))
    return config_dir / "drevalpy.json"


def load_config() -> DrevalConfig:
    """Read the user config file, returning defaults if it doesn't exist.

    :returns: Parsed config.
    """
    path = get_config_path()
    if not path.is_file():
        return DrevalConfig()
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return DrevalConfig.from_raw(raw)


def save_config(config: DrevalConfig) -> None:
    """Write the config to disk, creating the directory if needed.

    :param config: Config to persist.
    """
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_raw(), f, indent=2)
        f.write("\n")
