"""Plot requirement declarations: what a plot needs in an experiment, and the one check of them.

A plot declares the :class:`PlotRequirement` values it needs in ``Plot.requirements``. A plot is
built exactly when :func:`requirements_met` accepts them: the registry's ``applicable`` is the only
caller, and the report and ``save_all_png`` go through it, so a plot does not re-check what it
declares. A plot still handles what the declarations cannot express, such as models that share no
fold or a metric that is undefined everywhere.

The reference model of a normalized experiment is not among ``ExperimentResult.models``, but the
leaderboard and the critical difference diagram list it from ``reference_runs``, so it counts as a
model here.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult


class PlotRequirement(Enum):
    """Requirements that a plot class may declare."""

    #: Two or more models.
    MULTIPLE_MODELS = auto()
    #: Three or more models, the least the Friedman test of a critical difference diagram accepts.
    AT_LEAST_THREE_MODELS = auto()
    #: Two or more cross-validation folds.
    MULTIPLE_FOLDS = auto()
    #: Runs of a randomization test.
    RANDOMIZATION = auto()
    #: Runs of a robustness test.
    ROBUSTNESS = auto()


def requirements_met(experiment: ExperimentResult, requirements: frozenset[PlotRequirement]) -> bool:
    """Check whether an experiment has the data a set of plot requirements asks for.

    :param experiment: Experiment to check.
    :param requirements: Requirements to check.
    :returns: True if all requirements are met.
    """
    n_models = experiment.n_models + (1 if experiment.reference_runs else 0)
    available = {
        PlotRequirement.MULTIPLE_MODELS: n_models >= 2,
        PlotRequirement.AT_LEAST_THREE_MODELS: n_models >= 3,
        PlotRequirement.MULTIPLE_FOLDS: experiment.max_folds >= 2,
        PlotRequirement.RANDOMIZATION: experiment.has_randomization,
        PlotRequirement.ROBUSTNESS: experiment.has_robustness,
    }
    return all(available[requirement] for requirement in requirements)
