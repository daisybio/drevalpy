"""Tests for :mod:`drevalpy.experiment._tracking`.

The functions are stubs until W1-P6 lands. What is pinned here is the contract the
import-cost policy depends on: importing the module must not import ``wandb``.
"""

from __future__ import annotations

import inspect

import pytest

from drevalpy.experiment import _tracking


def test_importing_the_module_does_not_import_wandb() -> None:
    assert "wandb" not in vars(_tracking)


def test_tracking_run_signature() -> None:
    parameters = inspect.signature(_tracking.tracking_run).parameters

    assert list(parameters) == ["project", "name", "config", "tags"]
    assert parameters["name"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["config"].default is None
    assert parameters["tags"].default == ()


def test_stubs_are_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError, match="W1-P6"):
        _tracking.tracking_run(None, name="x")
    with pytest.raises(NotImplementedError, match="W1-P6"):
        _tracking.log_summary({"RMSE": 1.0})
