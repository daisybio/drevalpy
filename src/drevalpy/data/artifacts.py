"""Download and cache external model artifacts (PPI embeddings, checkpoints, etc.)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable

from upath import UPath as Path

from drevalpy.data._paths import get_default_data_dir
from drevalpy.data._transfer import download_file, download_files
from drevalpy.log import get_logger

logger = get_logger(__name__)

_DEFAULT_ARTIFACTS_URI = "s3://orakl-open-source-data/drevalpy/artifacts/"
_URI_ENV_VAR = "DREVALPY_ARTIFACTS_URI"
_STORAGE_OPTIONS_ENV_VAR = "DREVALPY_ARTIFACTS_STORAGE_OPTIONS"


def get_artifacts_uri() -> str:
    """Return the base URI artifacts are fetched from.

    Override with ``DREVALPY_ARTIFACTS_URI`` to point at a mirror, a local
    directory, or another fsspec-supported protocol.

    :returns: Base URI ending in a path separator.
    """
    return os.environ.get(_URI_ENV_VAR, "").strip() or _DEFAULT_ARTIFACTS_URI


def get_artifacts_storage_options() -> dict:
    """Return fsspec storage options for the artifacts location.

    Empty by default so that fsspec applies the ambient credential chain (env
    vars, shared config, EC2/ECS instance roles). Set
    ``DREVALPY_ARTIFACTS_STORAGE_OPTIONS`` to a JSON object to pass explicit
    options such as ``{"profile": "my-profile"}`` or ``{"anon": true}``.

    :returns: Mapping forwarded to the fsspec filesystem.
    """
    raw = os.environ.get(_STORAGE_OPTIONS_ENV_VAR, "").strip()
    if not raw:
        return {}
    try:
        options = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring %s: not valid JSON.", _STORAGE_OPTIONS_ENV_VAR)
        return {}
    if not isinstance(options, dict):
        logger.warning("Ignoring %s: expected a JSON object.", _STORAGE_OPTIONS_ENV_VAR)
        return {}
    return options


def _remote_root() -> Path:
    """Return the remote artifacts root as a configured UPath."""
    return Path(get_artifacts_uri(), **get_artifacts_storage_options())


def _cache_root() -> Path:
    """Return the local directory artifacts are cached in."""
    return get_default_data_dir() / "artifacts"


def get_artifact(name: str) -> Path:
    """Return the local path to a named artifact, downloading it if absent.

    Artifacts are cached under ``DREVALPY_CACHE_DIR / artifacts / <name>``.

    :param name: Filename of the artifact (e.g. ``"human_ppi_features.tsv"``).
    :returns: Local path to the cached artifact file.
    """
    local_path = _cache_root() / name
    if local_path.exists():
        return local_path

    logger.info("Downloading artifact %s ...", name)
    download_file(_remote_root() / name, local_path, name)
    logger.info("Cached artifact at %s", local_path)
    return local_path


def get_artifact_dir(name: str, filenames: Iterable[str]) -> Path:
    """Return the local path to a multi-file artifact, downloading it if absent.

    The artifact is considered cached only once every expected file is present,
    so a download interrupted midway is retried rather than silently reused.

    :param name: Directory name of the artifact within the artifacts location.
    :param filenames: Files the artifact directory must contain.
    :returns: Local path to the cached artifact directory.
    """
    expected = tuple(filenames)
    local_dir = _cache_root() / name
    if all((local_dir / filename).exists() for filename in expected):
        return local_dir

    logger.info("Downloading artifact directory %s ...", name)
    download_files(_remote_root() / name, local_dir, name, expected)
    logger.info("Cached artifact at %s", local_dir)
    return local_dir
