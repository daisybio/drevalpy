"""Download and cache the MSigDB gene sets Precily scores pathways with."""

from __future__ import annotations

from upath import UPath as Path

from drevalpy.data._paths import get_default_data_dir
from drevalpy.data._transfer import cached_file_matches, download_checked
from drevalpy.errors import InputNotFoundError
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
    checked against a pinned SHA-256: a cached file that differs is deleted and fetched again, and a
    download that differs raises ``InvalidInputError`` and is deleted.

    :returns: Local path to the cached GMT file.
    :raises InputNotFoundError: If the download fails; the message names the URL.
    """
    local = _cache_path()
    if cached_file_matches(local, GMT_SHA256):
        return local

    logger.info("Downloading MSigDB gene sets from %s ...", GMT_URL)
    try:
        download_checked(Path(GMT_URL), local, GMT_FILENAME, GMT_SHA256)
    except _download_failures() as exc:
        raise InputNotFoundError(
            f"Could not download the MSigDB gene sets from {GMT_URL}: {exc}. "
            f'Check the connection, or store the gene sets in the dataset as uns["pathways_gmt"].'
        ) from exc
    logger.info("Cached MSigDB gene sets at %s", local)
    return local
