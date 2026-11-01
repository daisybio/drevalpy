"""Streaming downloads with progress reporting for remote datasets and artifacts, and their SHA-256 check."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable

from rich.progress import DownloadColumn, Progress, TimeRemainingColumn, TransferSpeedColumn
from upath import UPath as Path

from drevalpy.errors import InvalidInputError
from drevalpy.log import get_logger

logger = get_logger(__name__)

_CHUNK_SIZE = 1024 * 256
_HASH_CHUNK_SIZE = 1024 * 1024
_hash_cache: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path) -> str:
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


def cached_file_matches(local_path: Path, sha256: str) -> bool:
    """Whether *local_path* is a file with the SHA-256 *sha256*; a file with another hash is deleted.

    :param local_path: Cache path of the file.
    :param sha256: Pinned hex digest, in either case.
    :returns: ``True`` if the cached file can be used, ``False`` if it must be downloaded.
    """
    if not local_path.is_file():
        return False
    if file_sha256(local_path) == sha256.lower():
        return True
    logger.info("Cached file at %s does not match the pinned sha256, removing and re-downloading.", local_path)
    local_path.unlink()
    return False


def download_checked(remote: Path, local_path: Path, label: str, sha256: str | None) -> Path:
    """Download a single remote file with :func:`download_file` and check it against a pinned SHA-256.

    :param remote: Remote source path (any fsspec-compatible protocol).
    :param local_path: Destination path on the local filesystem.
    :param label: Human-readable name shown in the progress bar and the error.
    :param sha256: Pinned hex digest, in either case, or ``None`` to skip the check.
    :returns: The local destination path.
    :raises InvalidInputError: If the downloaded file does not match *sha256*; the file is deleted.
    """
    download_file(remote, local_path, label)
    if sha256 is not None:
        actual = file_sha256(local_path)
        if actual != sha256.lower():
            local_path.unlink()
            raise InvalidInputError(
                f"The file downloaded for {label!r} from {remote} does not match the pinned sha256: "
                f"expected {sha256.lower()}, got {actual}. The file was deleted."
            )
    return local_path


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
