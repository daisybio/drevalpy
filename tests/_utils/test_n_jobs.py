"""Tests for ``check_n_jobs``."""

from __future__ import annotations

import pytest

from drevalpy._utils.n_jobs import check_n_jobs
from drevalpy.errors import InvalidInputError


@pytest.mark.parametrize("n_jobs", [1, 2, 16, -1])
def test_check_n_jobs_accepts_a_worker_count_or_all_cpus(n_jobs: int) -> None:
    check_n_jobs(n_jobs)


@pytest.mark.parametrize("n_jobs", [0, -2, -8])
def test_check_n_jobs_rejects_everything_else(n_jobs: int) -> None:
    with pytest.raises(InvalidInputError, match="n_jobs must be a positive integer or -1"):
        check_n_jobs(n_jobs)
