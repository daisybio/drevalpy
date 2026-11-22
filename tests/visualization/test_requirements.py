"""Which plots an experiment qualifies for: ``requirements_met`` against the data an experiment holds."""

from __future__ import annotations

import pytest

from drevalpy.visualization.requirements import PlotRequirement, requirements_met
from tests.synthetic import REFERENCE_MODEL, make_experiment_result


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


@pytest.mark.parametrize(("n_models", "expected"), [(2, False), (3, True)])
def test_at_least_three_models_is_the_floor_of_a_rank_test(n_models: int, expected: bool) -> None:
    experiment = make_experiment_result(n_models=n_models, n_folds=2)

    assert requirements_met(experiment, frozenset({PlotRequirement.AT_LEAST_THREE_MODELS})) is expected


@pytest.mark.parametrize(
    ("n_models", "requirement", "expected"),
    [
        (2, PlotRequirement.MULTIPLE_MODELS, True),
        (2, PlotRequirement.AT_LEAST_THREE_MODELS, False),
        (3, PlotRequirement.AT_LEAST_THREE_MODELS, True),
    ],
)
def test_the_reference_model_of_a_normalized_experiment_counts_as_a_model(
    n_models: int, requirement: PlotRequirement, expected: bool
) -> None:
    normalized = make_experiment_result(n_models=n_models, n_folds=2).normalize(REFERENCE_MODEL)

    assert normalized.n_models == n_models - 1
    assert requirements_met(normalized, frozenset({requirement})) is expected
