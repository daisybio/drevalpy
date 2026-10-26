"""Which plots an experiment qualifies for: ``requirements_met`` against the data an experiment holds."""

from __future__ import annotations

import pytest

from drevalpy.visualization.requirements import PlotRequirement, requirements_met
from tests.synthetic import make_experiment_result


def test_no_requirements_are_always_met() -> None:
    assert requirements_met(make_experiment_result(n_models=1, n_folds=1), frozenset()) is True


@pytest.mark.parametrize(
    ("requirement", "n_models", "n_folds"),
    [
        pytest.param(PlotRequirement.MULTIPLE_MODELS, 1, 2, id="one-model"),
        pytest.param(PlotRequirement.MULTIPLE_FOLDS, 2, 1, id="one-fold"),
    ],
)
def test_structural_requirements_are_rejected_when_unmet(
    requirement: PlotRequirement, n_models: int, n_folds: int
) -> None:
    experiment = make_experiment_result(n_models=n_models, n_folds=n_folds)

    assert requirements_met(experiment, frozenset({requirement})) is False


def test_structural_requirements_are_accepted_when_met() -> None:
    experiment = make_experiment_result(n_models=2, n_folds=2)

    requirements = frozenset({PlotRequirement.MULTIPLE_MODELS, PlotRequirement.MULTIPLE_FOLDS})
    assert requirements_met(experiment, requirements) is True


@pytest.mark.parametrize(
    ("requirement", "flag"),
    [(PlotRequirement.RANDOMIZATION, "with_randomization"), (PlotRequirement.ROBUSTNESS, "with_robustness")],
)
def test_data_requirements_need_the_matching_data(requirement: PlotRequirement, flag: str) -> None:
    requirements = frozenset({requirement})

    assert requirements_met(make_experiment_result(n_models=2, n_folds=2), requirements) is False
    assert requirements_met(make_experiment_result(n_models=2, n_folds=2, **{flag: True}), requirements) is True
