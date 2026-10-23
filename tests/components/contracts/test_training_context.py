"""Tests for the per-training-call runtime context.

``training_context`` stores its checkpoint directory as a ``pathlib.Path``
rather than the ``UPath`` used elsewhere in the package; these tests assert the
behaviour that exists today rather than the repo-wide convention.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import pytest

from drevalpy.components.contracts.training_context import TrainingContext, check_n_jobs
from drevalpy.errors import InvalidInputError


def test_training_context_default_checkpoint_dir_is_a_unique_temp_path_not_the_cwd() -> None:
    first, second = TrainingContext(), TrainingContext()

    assert first.checkpoint_dir != second.checkpoint_dir
    assert first.checkpoint_dir.is_absolute()
    assert Path(tempfile.gettempdir()) in Path(str(first.checkpoint_dir)).parents
    assert not first.checkpoint_dir.exists()


def test_training_context_defaults_to_empty_logging_metadata() -> None:
    context = TrainingContext()

    assert context.logging_metadata == {}


def test_training_context_logging_metadata_is_not_shared_between_instances() -> None:
    first = TrainingContext()
    second = TrainingContext()

    first.logging_metadata["run"] = "1"

    assert second.logging_metadata == {}


def test_training_context_accepts_an_explicit_checkpoint_dir(tmp_path) -> None:
    context = TrainingContext(checkpoint_dir=tmp_path, logging_metadata={"fold": "0"})

    assert context.checkpoint_dir == tmp_path
    assert context.logging_metadata == {"fold": "0"}


def test_training_context_is_frozen(tmp_path) -> None:
    context = TrainingContext(checkpoint_dir=tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        context.checkpoint_dir = tmp_path / "other"


def test_training_context_fits_in_one_process_by_default() -> None:
    assert TrainingContext().n_jobs == 1


@pytest.mark.parametrize("n_jobs", [1, 2, 16, -1])
def test_check_n_jobs_accepts_a_worker_count_or_all_cpus(n_jobs: int) -> None:
    check_n_jobs(n_jobs)


@pytest.mark.parametrize("n_jobs", [0, -2, -8])
def test_check_n_jobs_rejects_everything_else(n_jobs: int) -> None:
    with pytest.raises(InvalidInputError, match="n_jobs must be a positive integer or -1"):
        check_n_jobs(n_jobs)
