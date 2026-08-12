"""Tests for the per-training-call runtime context.

``training_context`` stores its checkpoint directory as a ``pathlib.Path``
rather than the ``UPath`` used elsewhere in the package; these tests assert the
behaviour that exists today rather than the repo-wide convention.
"""

from __future__ import annotations

import dataclasses

import pytest

from drevalpy.components.contracts.training_context import _DEFAULT_CHECKPOINT_DIR, TrainingContext


def test_training_context_defaults_to_the_module_checkpoint_dir() -> None:
    context = TrainingContext()

    assert context.checkpoint_dir == _DEFAULT_CHECKPOINT_DIR
    assert str(_DEFAULT_CHECKPOINT_DIR) == "checkpoints"


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
