"""Local cache handling for registered dataset ``.h5mu`` files.

Owns cache lookup, SHA-256 and corruption checks, cache invalidation, and downloading missing or
invalid registered datasets into the cache directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from upath import UPath as Path

from drevalpy.data._paths import get_default_data_dir
from drevalpy.data._transfer import cached_file_matches, download_file
from drevalpy.log import get_logger
from drevalpy.types.data.dataset import Dataset

if TYPE_CHECKING:
    from drevalpy.registry.dataset._models import DatasetEntry, SourceEntry

logger = get_logger(__name__)


def _is_corruption_error(exc: Exception) -> bool:
    """Return whether *exc* indicates unusable file content rather than an environment error."""
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return True
    return isinstance(exc, OSError) and not isinstance(exc, (PermissionError, FileNotFoundError))


def load_cached(path: Path, sha256: str | None = None) -> Dataset | None:
    """Load a cached ``.h5mu`` file, deleting it and returning ``None`` if corrupt or hash-mismatched.

    Only call this for files belonging to a registered dataset that can be re-downloaded.
    Permission and environment errors propagate without deleting the file.

    :param path: Local cache file path.
    :param sha256: Expected hex SHA-256, or ``None`` to skip verification.
    :returns: Loaded dataset, or ``None`` when the cached file was invalid and removed.
    :raises Exception: Any non-corruption error (such as permission or dependency failures), re-raised unchanged.
    """
    if sha256 is not None and not cached_file_matches(path, sha256):
        logger.debug("Skipping cached file %s: failed SHA-256 check", path)
        return None

    try:
        return Dataset.load(path)
    except Exception as exc:
        if not _is_corruption_error(exc):
            logger.warning("Failed to load cached file %s (%s); leaving file intact.", path, exc)
            raise
        logger.warning("Corrupted file at %s (%s), removing and re-downloading.", path, exc)
        path.unlink()
        return None


def fetch_registered(name: str, entry: DatasetEntry, source: SourceEntry) -> Dataset:
    """Load a registered dataset from cache, downloading it first if absent or invalid.

    :param name: Registered dataset name (used for progress and error messages).
    :param entry: Registry entry describing the file name and optional pinned SHA-256.
    :param source: Registry source describing the remote URL and storage options.
    :returns: Loaded dataset.
    """
    local_path = get_default_data_dir() / entry.file
    if local_path.is_file():
        dataset = load_cached(local_path, entry.sha256)
        if dataset is not None:
            logger.info("Using cached dataset '%s' from %s", name, local_path)
            return dataset
    else:
        logger.debug("Dataset '%s' not found in cache at %s", name, local_path)
    remote = Path(source.url, **source.storage_options) / entry.file
    logger.info("Downloading dataset '%s' from %s to %s", name, remote, local_path)
    downloaded = download_file(remote, local_path, name, sha256=entry.sha256)
    logger.info("Cached dataset '%s' at %s", name, downloaded)
    return Dataset.load(downloaded)
