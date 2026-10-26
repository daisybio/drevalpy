"""Tests for ``TrainingContext``."""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import pytest

from drevalpy.components.contracts.training_context import TrainingContext


def test_default_checkpoint_dir_is_a_unique_temp_path_not_the_cwd() -> None:
    first, second = TrainingContext(), TrainingContext()

    assert first.checkpoint_dir != second.checkpoint_dir
    assert first.checkpoint_dir.is_absolute()
    assert Path(tempfile.gettempdir()) in Path(str(first.checkpoint_dir)).parents
    assert not first.checkpoint_dir.exists()


def test_an_explicit_checkpoint_dir_is_kept_and_the_context_is_frozen(tmp_path) -> None:
    context = TrainingContext(checkpoint_dir=tmp_path)

    assert context.checkpoint_dir == tmp_path
    assert context.n_jobs == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.checkpoint_dir = tmp_path / "other"
