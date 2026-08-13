"""Stdlib-only memory probes and a one-line stage logger.

Report generation has historically died with a bare ``exit 137`` and no output,
so these helpers exist to make the last line before a SIGKILL name both the
stage and the memory headroom that was left. They deliberately avoid ``psutil``:
it is only a transitive ``wandb`` dependency and is not guaranteed to be present
in a report-only install.

Every filesystem path is a parameter so the readers can be pointed at fixtures
in tests rather than at the host's real ``/proc`` and cgroup trees.
"""

from __future__ import annotations

import logging
import resource
import sys

from upath import UPath

PROC_STATUS = "/proc/self/status"
CGROUP_V2_LIMIT = "/sys/fs/cgroup/memory.max"
CGROUP_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

_BYTES_PER_GB = 1024**3
_KIB_PER_GB = 1024**2

#: Above this the cgroup value is a sentinel for "unlimited" rather than a cap.
#: Unbounded cgroups report a value near 2**63, and some report ``PAGE_COUNTER_MAX``.
_UNLIMITED_GB = 1024.0 * 1024.0


def _rusage_max_rss_gb() -> float:
    """Peak RSS from :func:`resource.getrusage`, normalised across platforms.

    ``ru_maxrss`` is KiB on Linux but bytes on macOS.

    :returns: Peak resident set size of this process in GB.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = _BYTES_PER_GB if sys.platform == "darwin" else _KIB_PER_GB
    return raw / divisor


def _read_status_kib(field: str, status_path: str) -> float | None:
    """Read one ``VmXxx`` field, in kB, out of a ``/proc/<pid>/status`` file.

    :param field: Field name without the colon, e.g. ``"VmRSS"``.
    :param status_path: Path to the status file to parse.
    :returns: The value in kB, or ``None`` if the file or field is unavailable.
    """
    prefix = f"{field}:"
    try:
        text = UPath(status_path).read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(prefix):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    return None
    return None


def rss_gb(status_path: str = PROC_STATUS) -> float:
    """Current resident set size of this process.

    :param status_path: Path to a ``/proc/<pid>/status``-formatted file.
    :returns: Current RSS in GB, falling back to peak RSS where ``/proc`` is absent.
    """
    kib = _read_status_kib("VmRSS", status_path)
    if kib is None:
        return _rusage_max_rss_gb()
    return kib / _KIB_PER_GB


def peak_rss_gb(status_path: str = PROC_STATUS) -> float:
    """High-water mark of this process's resident set size.

    :param status_path: Path to a ``/proc/<pid>/status``-formatted file.
    :returns: Peak RSS in GB.
    """
    kib = _read_status_kib("VmHWM", status_path)
    if kib is None:
        return _rusage_max_rss_gb()
    return kib / _KIB_PER_GB


def memory_limit_gb(
    v2_path: str = CGROUP_V2_LIMIT,
    v1_path: str = CGROUP_V1_LIMIT,
) -> float | None:
    """Container memory cap as seen from inside a cgroup.

    Inside an AWS Batch container this reports the real allocation, which is what
    makes the headroom figure in :func:`log_stage` meaningful. Outside one it is
    expected to be unavailable.

    :param v2_path: cgroup v2 ``memory.max`` path.
    :param v1_path: cgroup v1 ``memory.limit_in_bytes`` path, tried second.
    :returns: The cap in GB, or ``None`` when unreadable, unparseable or unlimited.
    """
    for path in (v2_path, v1_path):
        try:
            raw = UPath(path).read_text().strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            limit = int(raw) / _BYTES_PER_GB
        except ValueError:
            continue
        if limit >= _UNLIMITED_GB:
            continue
        return limit
    return None


def format_stage(
    stage: str,
    *,
    status_path: str = PROC_STATUS,
    v2_path: str = CGROUP_V2_LIMIT,
    v1_path: str = CGROUP_V1_LIMIT,
) -> str:
    """Build the one-line memory summary for a stage.

    :param stage: Human-readable name of the stage being entered or left.
    :param status_path: Path to a ``/proc/<pid>/status``-formatted file.
    :param v2_path: cgroup v2 ``memory.max`` path.
    :param v1_path: cgroup v1 ``memory.limit_in_bytes`` path.
    :returns: A line such as ``"load | rss=1.42 GB peak=1.51 GB limit=36.00 GB (4%)"``.
    """
    rss = rss_gb(status_path)
    peak = peak_rss_gb(status_path)
    limit = memory_limit_gb(v2_path, v1_path)
    line = f"{stage} | rss={rss:.2f} GB peak={peak:.2f} GB"
    if limit is not None and limit > 0:
        line += f" limit={limit:.2f} GB ({peak / limit:.0%})"
    return line


def log_stage(
    logger: logging.Logger,
    stage: str,
    *,
    level: int = logging.INFO,
    status_path: str = PROC_STATUS,
    v2_path: str = CGROUP_V2_LIMIT,
    v1_path: str = CGROUP_V1_LIMIT,
) -> None:
    """Emit :func:`format_stage` through ``logger``.

    :param logger: Logger to emit through.
    :param stage: Human-readable name of the stage being entered or left.
    :param level: Logging level for the emitted record.
    :param status_path: Path to a ``/proc/<pid>/status``-formatted file.
    :param v2_path: cgroup v2 ``memory.max`` path.
    :param v1_path: cgroup v1 ``memory.limit_in_bytes`` path.
    """
    logger.log(
        level,
        "%s",
        format_stage(stage, status_path=status_path, v2_path=v2_path, v1_path=v1_path),
    )
