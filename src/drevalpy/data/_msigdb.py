"""Download and cache the MSigDB gene sets Precily scores pathways with."""

from __future__ import annotations

from upath import UPath as Path

from drevalpy.data._paths import get_default_data_dir
from drevalpy.data._transfer import download_file
from drevalpy.errors import InputNotFoundError, InvalidInputError
from drevalpy.log import get_logger

logger = get_logger(__name__)

GMT_FILENAME = "c2.cp.v6.1.symbols.gmt"
GMT_URL = f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/6.1/{GMT_FILENAME}"
GMT_SHA256 = "ab98c58451aa648af402ecb1ab6701e03c9490753dd12272f6a2bd004df87f51"


def _cache_path() -> Path:
    """Return where the gene sets are cached."""
    return get_default_data_dir() / "msigdb" / GMT_FILENAME


def _download_failures() -> tuple[type[Exception], ...]:
    """Return the exception types a failed HTTP download can raise."""
    import aiohttp

    return (OSError, aiohttp.ClientError)


def get_precily_gmt() -> Path:
    """Return the local path to the MSigDB C2 CP v6.1 gene sets, downloading them on first use.

    The file ``c2.cp.v6.1.symbols.gmt`` is cached under ``DREVALPY_CACHE_DIR / msigdb``. It is
    checked against a pinned SHA-256: a cached file that differs is deleted and fetched again.

    :returns: Local path to the cached GMT file.
    :raises InputNotFoundError: If the download fails; the message names the URL.
    :raises InvalidInputError: If the downloaded file does not match the pinned SHA-256; the file is
        deleted.
    """
    from drevalpy.data.datasets._load import _file_sha256

    local = _cache_path()
    if local.is_file():
        if _file_sha256(local) == GMT_SHA256:
            return local
        logger.info("Cached gene sets at %s do not match the pinned sha256, removing and re-downloading.", local)
        local.unlink()

    logger.info("Downloading MSigDB gene sets from %s ...", GMT_URL)
    try:
        download_file(Path(GMT_URL), local, GMT_FILENAME)
    except _download_failures() as exc:
        raise InputNotFoundError(
            f"Could not download the MSigDB gene sets from {GMT_URL}: {exc}. "
            f'Check the connection, or store the gene sets in the dataset as uns["pathways_gmt"].'
        ) from exc
    actual = _file_sha256(local)
    if actual != GMT_SHA256:
        local.unlink()
        raise InvalidInputError(
            f"The MSigDB gene sets downloaded from {GMT_URL} do not match the pinned sha256: "
            f"expected {GMT_SHA256}, got {actual}. The file was deleted."
        )
    logger.info("Cached MSigDB gene sets at %s", local)
    return local
