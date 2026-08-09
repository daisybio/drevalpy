"""Load MuDatasets with protocol-agnostic downloads via universal-pathlib."""

from __future__ import annotations

from upath import UPath as Path

from drevalpy.log import get_logger

from ._paths import get_default_data_dir, resolve_h5mu_path
from .dataset_registry import registry
from .structures.mudataset import MuDataset

logger = get_logger(__name__)


def _download_h5mu(name: str) -> Path:
    """Download the .h5mu file for a registered dataset.

    :param name: Dataset name from the registry.
    :returns: Local path to the downloaded file.
    """
    from rich.progress import DownloadColumn, Progress, TimeRemainingColumn, TransferSpeedColumn

    entry = registry.datasets[name]
    source = registry.sources[entry.source]
    remote = Path(source.url, **source.storage_options) / entry.file
    local_path = get_default_data_dir() / entry.file
    local_path.parent.mkdir(parents=True, exist_ok=True)

    fs = remote.fs
    remote_key = remote.path
    size = fs.size(remote_key)

    with Progress(
        *Progress.get_default_columns(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"Downloading {name}", total=size)

        with fs.open(remote_key, "rb", block_size=0) as src, open(local_path, "wb") as dst:
            while chunk := src.read(1024 * 256):
                dst.write(chunk)
                progress.advance(task, len(chunk))

    return local_path


def load_mudataset(dataset_name: str) -> MuDataset:
    """Load a registered or custom dataset as a MuDataset from its .h5mu file.

    Resolution order:

    1. If the .h5mu exists at the standard cache path, load it directly.
    2. If *dataset_name* is registered, download the .h5mu if needed and load.
    3. If *dataset_name* is a path to an existing .h5mu file, load it directly.

    :param dataset_name: Registered dataset name, or path to a .h5mu file.
    :returns: Loaded MuDataset.
    :raises FileNotFoundError: If the .h5mu file cannot be found or downloaded.
    """
    h5mu_path = resolve_h5mu_path(dataset_name)
    if h5mu_path.is_file():
        try:
            return MuDataset.from_file(h5mu_path)
        except Exception:
            logger.warning("Corrupted file at %s, removing and re-downloading.", h5mu_path)
            h5mu_path.unlink()

    if registry.is_registered(dataset_name):
        entry = registry.datasets[dataset_name]
        data_dir = get_default_data_dir()
        candidate = data_dir / entry.file
        if candidate.is_file():
            try:
                return MuDataset.from_file(candidate)
            except Exception:
                logger.warning("Corrupted file at %s, removing and re-downloading.", candidate)
                candidate.unlink()
        downloaded = _download_h5mu(dataset_name)
        return MuDataset.from_file(downloaded)

    candidate_path = Path(dataset_name)
    if candidate_path.is_file() and candidate_path.suffix == ".h5mu":
        return MuDataset.from_file(candidate_path)

    raise FileNotFoundError(
        f"Cannot locate .h5mu for dataset '{dataset_name}'. "
        f"Checked: {h5mu_path}, registry ({registry.dataset_names}), and direct path."
    )
