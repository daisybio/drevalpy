"""Load datasets with protocol-agnostic downloads via universal-pathlib."""

from __future__ import annotations

import hashlib

from upath import UPath as Path

from drevalpy.data._paths import get_default_data_dir, resolve_h5mu_path
from drevalpy.data._transfer import download_file
from drevalpy.errors import InputNotFoundError, InvalidInputError
from drevalpy.log import get_logger
from drevalpy.registry.dataset._registry import dataset_registry
from drevalpy.types.data.dataset import Dataset

logger = get_logger(__name__)

_HASH_CHUNK_SIZE = 1024 * 1024
_hash_cache: dict[tuple[str, int, int], str] = {}


def _file_sha256(path: Path) -> str:
    """Return the hex SHA-256 of *path*, reading it in chunks.

    The digest is remembered for the rest of the process, keyed by path, size and
    modification time, so a file that is replaced on disk is hashed again.

    :param path: Local file to hash.
    :returns: Lowercase hex digest.
    """
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _hash_cache.get(key)
    if cached is None:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
        cached = _hash_cache[key] = digest.hexdigest()
    return cached


def _download(name: str) -> Path:
    """Download the .h5mu file for a registered dataset.

    When the registry entry carries a ``sha256``, the downloaded file is checked against it.

    :param name: Dataset name from the registry.
    :returns: Local path to the downloaded file.
    :raises InvalidInputError: If the downloaded file does not match the registered ``sha256``;
        the file is deleted.
    """
    entry = dataset_registry.datasets[name]
    source = dataset_registry.sources[entry.source]
    remote = Path(source.url, **source.storage_options) / entry.file
    local = download_file(remote, get_default_data_dir() / entry.file, name)
    if entry.sha256 is not None:
        actual = _file_sha256(local)
        if actual != entry.sha256.lower():
            local.unlink()
            raise InvalidInputError(
                f"Downloaded file for dataset '{name}' from {remote} does not match the registered sha256: "
                f"expected {entry.sha256.lower()}, got {actual}. The file was deleted."
            )
    return local


def _carries_curve_quality(dataset: Dataset) -> bool:
    """Whether the default curve-quality thresholds can be evaluated on *dataset*.

    The CurveCurator refit of the screens kept the file names of the generation
    before it, so a cache filled by an older drevalpy holds a file that parses
    perfectly but has none of the quality layers every splitter now reads. Left
    alone that surfaces as a ``KeyError`` from deep inside a split, so the loader
    treats such a file as stale and fetches it again.

    Asking the filter itself, rather than checking a list of layer names, keeps
    this honest when the default rule changes.
    """
    from drevalpy.data.quality import curve_quality_mask

    try:
        curve_quality_mask(dataset)
    except KeyError:
        return False
    return True


def _is_corruption_error(exc: Exception) -> bool:
    """Whether *exc* says the file content is unusable, as opposed to the environment failing."""
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return True
    return isinstance(exc, OSError) and not isinstance(exc, (PermissionError, FileNotFoundError))


def _load_from_cache(path: Path, sha256: str | None = None) -> Dataset | None:
    """Load a cached .h5mu, or return ``None`` when it must be fetched again.

    A file whose SHA-256 differs from *sha256* (when given) is deleted without being parsed.
    Otherwise the file is deleted only when parsing fails with a corruption-type error
    (``ValueError``, ``TypeError``, ``KeyError``, or an ``OSError`` other than
    ``PermissionError`` / ``FileNotFoundError``). Any other failure, such as missing
    permissions or a missing dependency, propagates and leaves the file alone.

    :param path: Local candidate path.
    :param sha256: Hex SHA-256 the file must have, or ``None`` to skip the check.
    :returns: The dataset, or ``None`` if the file was unusable and removed.
    :raises Exception: Any error that is not a corruption-type error, re-raised unchanged.
    """
    if sha256 is not None and _file_sha256(path) != sha256.lower():
        logger.info("Cached file at %s does not match the registered sha256, removing and re-downloading.", path)
        path.unlink()
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
    """Load a registered dataset from its cache file, downloading it when the cache is unusable.

    :param dataset_name: Registered dataset name.
    :param registered_path: Cache path of the registered file.
    :param sha256: Expected hash of the file, if the registry pins one.
    :returns: Loaded Dataset.
    """
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
        f"Checked: {h5mu_path}, registry ({dataset_registry.dataset_names}), and direct path."
    )
