"""Load datasets by file path or registered name."""

from __future__ import annotations

from upath import UPath as Path

from drevalpy.data.datasets._cache import fetch_registered
from drevalpy.errors import InputNotFoundError
from drevalpy.log import get_logger
from drevalpy.registry.dataset._registry import dataset_registry
from drevalpy.types.data.dataset import Dataset

logger = get_logger(__name__)


def load(dataset_name: str) -> Dataset:
    """Load a dataset as a :class:`Dataset` from a ``.h5mu`` path or a registered name.

    Resolution order:

    1. If *dataset_name* is a path to an existing ``.h5mu`` file, load it directly.
    2. If *dataset_name* is registered, load it from the cache directory (downloading it if
       missing, corrupted, or mismatched against its pinned SHA-256).

    Direct paths are always taken at face value and never deleted.

    :param dataset_name: Path to a ``.h5mu`` file or a registered dataset name.
    :returns: Loaded Dataset.
    :raises InputNotFoundError: If the ``.h5mu`` file cannot be found or downloaded.
    """
    candidate = Path(dataset_name)
    if candidate.suffix == ".h5mu" and candidate.is_file():
        logger.info("Loading dataset from file %s", candidate)
        return Dataset.load(candidate)

    entry = dataset_registry.datasets.get(dataset_name)
    if entry is not None:
        return fetch_registered(dataset_name, entry, dataset_registry.sources[entry.source])

    logger.debug("Dataset %r not found as a local .h5mu file or in registry", dataset_name)
    raise InputNotFoundError(
        f"Cannot locate .h5mu for dataset '{dataset_name}'. "
        f"Checked direct path and registry ({list(dataset_registry.names())})."
    )
