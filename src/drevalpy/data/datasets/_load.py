"""Load datasets with protocol-agnostic downloads via universal-pathlib."""

from __future__ import annotations

from upath import UPath as Path

from drevalpy.data._paths import get_default_data_dir, resolve_h5mu_path
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


def load(dataset_name: str) -> Dataset:
    """Load a registered or custom dataset as a Dataset from its .h5mu file.

    Resolution order:

    1. If the .h5mu exists at the standard cache path, load it directly.
    2. If *dataset_name* is registered, download the .h5mu if needed and load.
    3. If *dataset_name* is a path to an existing .h5mu file, load it directly.

    A cached file is re-downloaded when it cannot be parsed, when it is too
    old to carry the curve-quality layers the splitters filter on, or when the
    registry entry has a ``sha256`` that the file does not match. A download that
    does not match it raises ``InvalidInputError``. An explicit path in case 3 is
    always taken at face value.

    :param dataset_name: Registered dataset name, or path to a .h5mu file.
    :returns: Loaded Dataset.
    :raises InputNotFoundError: If the .h5mu file cannot be found or downloaded.
    """
    h5mu_path = resolve_h5mu_path(dataset_name)
    entry = dataset_registry.datasets.get(dataset_name)
    registered_path = get_default_data_dir() / entry.file if entry is not None else None

    if h5mu_path.is_file():
        expected = entry.sha256 if registered_path == h5mu_path and entry is not None else None
        dataset = _load_from_cache(h5mu_path, expected)
        if dataset is not None:
            return dataset

    if entry is not None and registered_path is not None:
        return _load_registered(dataset_name, registered_path, entry.sha256)

    candidate_path = Path(dataset_name)
    if candidate_path.is_file() and candidate_path.suffix == ".h5mu":
        return Dataset.load(candidate_path)

    raise InputNotFoundError(
        f"Cannot locate .h5mu for dataset '{dataset_name}'. "
        f"Checked: {h5mu_path}, registry ({list(dataset_registry.names())}), and direct path."
    )
