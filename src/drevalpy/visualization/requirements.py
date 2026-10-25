"""Plot requirement declarations for capability-based plot selection."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult


class PlotRequirement(Enum):
    """Requirements that a plot class may declare."""

    MULTIPLE_MODELS = auto()
    MULTIPLE_FOLDS = auto()
    RANDOMIZATION = auto()
    ROBUSTNESS = auto()


def requirements_met(experiment: ExperimentResult, requirements: frozenset[PlotRequirement]) -> bool:
    """Check whether an experiment has the data a set of plot requirements asks for.

    :param experiment: Experiment to check.
    :param requirements: Requirements to check.
    :returns: True if all requirements are met.
    """
    available = {
        PlotRequirement.MULTIPLE_MODELS: experiment.n_models >= 2,
        PlotRequirement.MULTIPLE_FOLDS: experiment.max_folds >= 2,
        PlotRequirement.RANDOMIZATION: experiment.has_randomization,
        PlotRequirement.ROBUSTNESS: experiment.has_robustness,
    }
    return all(available[requirement] for requirement in requirements)
