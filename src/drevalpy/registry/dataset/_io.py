"""Config file I/O with file locking for drevalpy dataset registry."""

from __future__ import annotations

import json

from filelock import FileLock
from upath import UPath as Path

from ._models import DrevalConfig

_LOCK_TIMEOUT = 10


def get_config_path() -> Path:
    """Return path to the dataset registry config file.

    :returns: Path to ``datasets.json`` in the config directory.
    """
    from drevalpy.data._paths import get_config_dir

    return get_config_dir() / "datasets.json"


def config_lock() -> FileLock:
    """Return the exclusive file lock for config read-modify-write operations.

    Use it as a context manager. The lock file is a sibling of the config file.
    ``is_singleton=True`` makes the lock reentrant: nesting two ``config_lock()``
    blocks in the same process reuses one underlying lock and its recursion
    counter, instead of deadlocking a second independent ``FileLock`` until the
    timeout expires.

    :returns: The lock, held for the duration of a ``with`` block.
    """
    return FileLock(get_config_path().with_suffix(".lock"), timeout=_LOCK_TIMEOUT, is_singleton=True)


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
