"""Memory probes and a one-line stage logger for the report.

Report generation can die with a bare ``exit 137`` and no output, so the last line before a
SIGKILL names the stage and the memory in use. The probes are stdlib-only: ``psutil`` is not a
dependency of a report-only install.
"""

from __future__ import annotations

import logging
import sys

from upath import UPath

PROC_STATUS = "/proc/self/status"


def _peak_rss_gb() -> float:
    """Peak RSS from :func:`resource.getrusage`: KiB on Linux, bytes on macOS, 0 on Windows (no ``resource``)."""
    try:
        import resource
    except ImportError:
        return 0.0
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024**3 if sys.platform == "darwin" else 1024**2)


def rss_gb(status_path: str = PROC_STATUS) -> float:
    """Read the current resident set size of this process.

    :param status_path: Path to a ``/proc/<pid>/status``-formatted file.
    :returns: Current RSS (``VmRSS``) in GB, or the peak RSS where the file or field is unavailable.
    """
    try:
        lines = UPath(status_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return _peak_rss_gb()
    for line in lines:
        name, _, value = line.partition(":")
        if name == "VmRSS":
            try:
                return float(value.split()[0]) / 1024**2
            except (IndexError, ValueError):
                break
    return _peak_rss_gb()


def log_stage(logger: logging.Logger, stage: str, *, level: int = logging.INFO) -> None:
    """Log ``"<stage> | rss=<current> GB peak=<peak> GB"`` through *logger*.

    :param logger: Logger to emit through.
    :param stage: Name of the stage being entered or left.
    :param level: Logging level of the record.
    """
    logger.log(level, "%s | rss=%.2f GB peak=%.2f GB", stage, rss_gb(), _peak_rss_gb())
