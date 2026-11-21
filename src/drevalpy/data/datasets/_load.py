"""Load datasets with protocol-agnostic downloads via universal-pathlib."""

from __future__ import annotations

from upath import UPath as Path

from drevalpy.data._paths import get_default_data_dir
from drevalpy.data._transfer import cached_file_matches, download_file
from drevalpy.errors import InputNotFoundError
from drevalpy.log import get_logger
from drevalpy.registry.dataset._registry import dataset_registry
from drevalpy.types.data.dataset import Dataset

logger = get_logger(__name__)


def _download(name: str) -> Path:
    """Download the .h5mu file for a registered dataset, verifying its pinned SHA-256."""
    entry = dataset_registry.datasets[name]
    source = dataset_registry.sources[entry.source]
    remote = Path(source.url, **source.storage_options) / entry.file
    return download_file(remote, get_default_data_dir() / entry.file, name, sha256=entry.sha256)


def _carries_curve_quality(dataset: Dataset) -> bool:
    """Return whether the default curve-quality thresholds can be evaluated on *dataset*."""
    from drevalpy.data.quality import CurveQuality

    try:
        CurveQuality().mask(dataset)
    except KeyError:
        return False
    return True


def _is_corruption_error(exc: Exception) -> bool:
    """Return whether *exc* indicates unusable file content rather than an environment error."""
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return True
    return isinstance(exc, OSError) and not isinstance(exc, (PermissionError, FileNotFoundError))


def _load_from_cache(path: Path, sha256: str | None = None) -> Dataset | None:
    """Load a cached .h5mu, deleting it and returning ``None`` if corrupted, stale, or mismatched."""
    if sha256 is not None and not cached_file_matches(path, sha256):
        return None

    try:
        dataset = Dataset.load(path)
    except Exception as exc:
        if not _is_corruption_error(exc):
            raise
        logger.warning("Corrupted file at %s (%s), removing and re-downloading.", path, exc)
        path.unlink()
        return None

    if not _carries_curve_quality(dataset):
        logger.warning(
            "Cached file at %s predates the curve-quality layers, removing and re-downloading.",
            path,
        )
        path.unlink()
        return None

    return dataset


def _load_registered(dataset_name: str, registered_path: Path, sha256: str | None) -> Dataset:
    """Load a registered dataset from cache, downloading it if absent or unusable."""
    if registered_path.is_file():
        dataset = _load_from_cache(registered_path, sha256)
        if dataset is not None:
            return dataset
    return Dataset.load(_download(dataset_name))


def _is_plain_name(value: str) -> bool:
    """Return whether *value* is a bare dataset name (not a path or URI)."""
    return bool(value) and not value.endswith(".h5mu") and "/" not in value and "\\" not in value and ":" not in value


def load(dataset_name: str) -> Dataset:
    """Load a registered or custom dataset as a Dataset from its .h5mu file.

    Resolution order:

    1. If *dataset_name* is a path to an existing ``.h5mu`` file, load it directly.
    2. If *dataset_name* is registered, load it from the cache directory (downloading it if
       missing, corrupted, lacking curve-quality layers, or mismatched against its pinned SHA-256).
    3. If *dataset_name* is a bare name matching ``<cache dir>/<name>.h5mu``, load it directly.

    Only cached files belonging to a registered dataset are ever deleted and re-downloaded; direct
    paths and unregistered files are always left untouched.

    :param dataset_name: Path to a ``.h5mu`` file or a registered/cached dataset name.
    :returns: Loaded Dataset.
    :raises InputNotFoundError: If the ``.h5mu`` file cannot be found or downloaded.
    """
    candidate = Path(dataset_name)
    if candidate.suffix == ".h5mu" and candidate.is_file():
        return Dataset.load(candidate)

    entry = dataset_registry.datasets.get(dataset_name)
    if entry is not None:
        return _load_registered(dataset_name, get_default_data_dir() / entry.file, entry.sha256)

    if _is_plain_name(dataset_name):
        cached = get_default_data_dir() / f"{dataset_name}.h5mu"
        if cached.is_file():
            return Dataset.load(cached)

    raise InputNotFoundError(
        f"Cannot locate .h5mu for dataset '{dataset_name}'. "
        f"Checked direct path, registry ({list(dataset_registry.names())}), and {get_default_data_dir()}."
    )
