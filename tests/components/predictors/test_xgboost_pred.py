"""Tests for the macOS-only XGBoost thread defaults in ``xgboost_pred.py``."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from drevalpy.components.predictors import xgboost_pred
from drevalpy.components.predictors.xgboost_pred import _set_xgboost_thread_defaults

THREAD_VARIABLES = tuple(xgboost_pred._XGBOOST_THREAD_ENV_DEFAULTS)


@pytest.fixture
def cleared_thread_variables(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove the four thread variables from the environment for one test."""
    for name in THREAD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_thread_defaults_set_nothing_off_macos(cleared_thread_variables: pytest.MonkeyPatch) -> None:
    cleared_thread_variables.setattr(sys, "platform", "linux")
    _set_xgboost_thread_defaults()
    assert {name: os.environ.get(name) for name in THREAD_VARIABLES} == dict.fromkeys(THREAD_VARIABLES)


def test_thread_defaults_set_all_four_on_macos(cleared_thread_variables: pytest.MonkeyPatch) -> None:
    cleared_thread_variables.setattr(sys, "platform", "darwin")
    _set_xgboost_thread_defaults()
    assert {name: os.environ.get(name) for name in THREAD_VARIABLES} == dict.fromkeys(THREAD_VARIABLES, "1")


def test_thread_defaults_keep_a_user_value_on_macos(cleared_thread_variables: pytest.MonkeyPatch) -> None:
    cleared_thread_variables.setattr(sys, "platform", "darwin")
    cleared_thread_variables.setenv("OMP_NUM_THREADS", "4")
    _set_xgboost_thread_defaults()
    assert os.environ["OMP_NUM_THREADS"] == "4"


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "darwin", reason="macOS sets the thread defaults on import")
def test_import_drevalpy_leaves_omp_num_threads_unset() -> None:
    env = {name: value for name, value in os.environ.items() if name != "OMP_NUM_THREADS"}
    completed = subprocess.run(
        [sys.executable, "-c", "import os, drevalpy; print(os.environ.get('OMP_NUM_THREADS'))"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "None"
