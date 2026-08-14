"""Load MuDatasets with protocol-agnostic downloads via universal-pathlib."""

from __future__ import annotations

from upath import UPath as Path

from drevalpy.data._paths import get_default_data_dir, resolve_h5mu_path
from drevalpy.data._transfer import download_file
from drevalpy.log import get_logger
from drevalpy.registry.dataset._registry import dataset_registry
from drevalpy.types.data.dataset import Dataset

logger = get_logger(__name__)


def _download(name: str) -> Path:
    """Download the .h5mu file for a registered dataset.

    :param name: Dataset name from the registry.
    :returns: Local path to the downloaded file.
    """
    entry = dataset_registry.datasets[name]
    source = dataset_registry.sources[entry.source]
    remote = Path(source.url, **source.storage_options) / entry.file
    return download_file(remote, get_default_data_dir() / entry.file, name)


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


def _load_from_cache(path: Path) -> Dataset | None:
    """Load a cached .h5mu, or return ``None`` when it must be fetched again.

    :param path: Local candidate path.
    :returns: The dataset, or ``None`` if the file was unusable and removed.
    """
    try:
        dataset = Dataset.load(path)
    except Exception:
        logger.warning("Corrupted file at %s, removing and re-downloading.", path)
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


def load(dataset_name: str) -> Dataset:
    """Load a registered or custom dataset as a Dataset from its .h5mu file.

    Resolution order:

    1. If the .h5mu exists at the standard cache path, load it directly.
    2. If *dataset_name* is registered, download the .h5mu if needed and load.
    3. If *dataset_name* is a path to an existing .h5mu file, load it directly.

    A cached file is re-downloaded when it cannot be parsed, or when it is too
    old to carry the curve-quality layers the splitters filter on. An explicit
    path in case 3 is always taken at face value.

    :param dataset_name: Registered dataset name, or path to a .h5mu file.
    :returns: Loaded Dataset.
    :raises FileNotFoundError: If the .h5mu file cannot be found or downloaded.
    """
    h5mu_path = resolve_h5mu_path(dataset_name)
    if h5mu_path.is_file():
        dataset = _load_from_cache(h5mu_path)
        if dataset is not None:
            return dataset

    if dataset_registry.is_registered(dataset_name):
        entry = dataset_registry.datasets[dataset_name]
        candidate = get_default_data_dir() / entry.file
        if candidate.is_file():
            dataset = _load_from_cache(candidate)
            if dataset is not None:
                return dataset
        downloaded = _download(dataset_name)
        return Dataset.load(downloaded)

    candidate_path = Path(dataset_name)
    if candidate_path.is_file() and candidate_path.suffix == ".h5mu":
        return Dataset.load(candidate_path)

    raise FileNotFoundError(
        f"Cannot locate .h5mu for dataset '{dataset_name}'. "
        f"Checked: {h5mu_path}, registry ({dataset_registry.dataset_names}), and direct path."
    )
