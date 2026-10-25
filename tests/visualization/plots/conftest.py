"""Fixtures shared by the plot tests."""

from __future__ import annotations

import pytest

from drevalpy.types.results.experiment import ExperimentResult
from tests.synthetic import make_experiment_result


@pytest.fixture(scope="module")
def experiment() -> ExperimentResult:
    """The default synthetic experiment, built once per module."""
    return make_experiment_result()
