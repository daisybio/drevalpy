"""Config file I/O with file locking for drevalpy dataset registry."""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager

from filelock import FileLock
from upath import UPath as Path

from drevalpy.data._paths import get_config_dir

from .models import DrevalConfig

_LOCK_TIMEOUT = 10


def get_config_path() -> Path:
    """Return path to the dataset registry config file.

    :returns: Path to ``datasets.json`` in the config directory.
    """
    return get_config_dir() / "datasets.json"


def _lock_path() -> Path:
    """Return the lock file path (sibling of the config file)."""
    return get_config_path().with_suffix(".lock")


@contextmanager
def config_lock() -> Generator[None, None, None]:
    """Acquire an exclusive file lock for config read-modify-write operations.

    :yields: Nothing; the lock is held for the duration of the context.
    """
    lock = FileLock(_lock_path(), timeout=_LOCK_TIMEOUT)
    with lock:
        yield


def load_config() -> DrevalConfig:
    """Read the user config file, returning defaults if it doesn't exist.

    Does NOT acquire the lock -- callers performing read-modify-write should
    use ``config_lock()`` to wrap the entire operation.

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

    Does NOT acquire the lock -- callers should wrap with ``config_lock()``.

    :param config: Config to persist.
    """
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_raw(), f, indent=2)
        f.write("\n")
