"""Download and cache external model artifacts (PPI embeddings, checkpoints, etc.)."""

from __future__ import annotations

from rich.progress import DownloadColumn, Progress, TimeRemainingColumn, TransferSpeedColumn
from upath import UPath as Path

from drevalpy.data._paths import get_default_data_dir
from drevalpy.log import get_logger

logger = get_logger(__name__)

_ARTIFACTS_BUCKET = "s3://omics-representation/drevalpy_artifacts/"
_STORAGE_OPTIONS = {"profile": "orakl"}


def get_artifact(name: str) -> Path:
    """Return the local path to a named artifact, downloading from S3 if absent.

    Artifacts are cached under ``DREVALPY_CACHE_DIR / artifacts / <name>``.

    :param name: Filename of the artifact (e.g. ``"human_ppi_features.tsv"``).
    :returns: Local path to the cached artifact file.
    """
    cache_dir = get_default_data_dir() / "artifacts"
    local_path = cache_dir / name
    if local_path.exists():
        return local_path

    logger.info("Downloading artifact %s ...", name)
    remote = Path(_ARTIFACTS_BUCKET, **_STORAGE_OPTIONS) / name
    _download_artifact(remote, local_path, name)
    return local_path


def _download_artifact(remote: Path, local_path: Path, name: str) -> None:
    """Download an artifact from S3 with a progress bar."""
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

    logger.info("Cached artifact at %s", local_path)
