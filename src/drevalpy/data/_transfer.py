"""Streaming downloads with progress reporting for remote datasets and artifacts."""

from __future__ import annotations

import os
from collections.abc import Iterable

from rich.progress import DownloadColumn, Progress, TimeRemainingColumn, TransferSpeedColumn
from upath import UPath as Path

from drevalpy.log import get_logger

logger = get_logger(__name__)

_CHUNK_SIZE = 1024 * 256


def download_file(remote: Path, local_path: Path, label: str) -> Path:
    """Download a single remote file to a local path.

    :param remote: Remote source path (any fsspec-compatible protocol).
    :param local_path: Destination path on the local filesystem.
    :param label: Human-readable name shown in the progress bar.
    :returns: The local destination path.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with _progress() as progress:
        _stream(remote, local_path, label, progress)
    return local_path


def download_files(remote_dir: Path, local_dir: Path, label: str, filenames: Iterable[str]) -> Path:
    """Download several files from a remote directory into a local directory.

    :param remote_dir: Remote directory holding the files.
    :param local_dir: Local destination directory (created if absent).
    :param label: Human-readable name shown in the progress bar.
    :param filenames: Names of the files to fetch from *remote_dir*.
    :returns: The local destination directory.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    with _progress() as progress:
        for name in filenames:
            _stream(remote_dir / name, local_dir / name, f"{label}/{name}", progress)
    return local_dir


def _progress() -> Progress:
    """Build a progress bar with transfer size, speed and ETA columns."""
    return Progress(
        *Progress.get_default_columns(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )


def _stream(remote: Path, local_path: Path, label: str, progress: Progress) -> None:
    """Copy one remote file to *local_path*, staging it under a unique ``.part`` file.

    Staging keeps concurrent readers from ever observing a truncated file, which
    matters when many workers share a cache directory.
    """
    fs = remote.fs
    remote_key = remote.path
    task = progress.add_task(f"Downloading {label}", total=fs.size(remote_key))

    partial = local_path.with_name(f"{local_path.name}.{os.getpid()}.part")
    try:
        with fs.open(remote_key, "rb", block_size=0) as src, open(partial, "wb") as dst:
            while chunk := src.read(_CHUNK_SIZE):
                dst.write(chunk)
                progress.advance(task, len(chunk))
        os.replace(partial, local_path)
    finally:
        partial.unlink(missing_ok=True)
