"""The one check of a requested worker count."""

from __future__ import annotations

from drevalpy.errors import InvalidInputError


def check_n_jobs(n_jobs: int) -> None:
    """Reject a worker count that no fit can use.

    :param n_jobs: Requested number of worker processes.
    :raises InvalidInputError: If it is neither positive nor ``-1`` (all CPUs).
    """
    if n_jobs < 1 and n_jobs != -1:
        raise InvalidInputError(f"n_jobs must be a positive integer or -1 (all CPUs), got {n_jobs}")
