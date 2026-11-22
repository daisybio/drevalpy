"""Plot requirement declarations: what a plot needs in an experiment, and the one check of them.

A plot declares the :class:`PlotRequirement` values it needs in ``Plot.requirements``. A plot is
built exactly when :func:`requirements_met` accepts them: the registry's ``applicable`` is the only
caller, and the report and ``save_all_png`` go through it, so a plot does not re-check what it
declares. A plot still handles what the declarations cannot express, such as models that share no
fold or a metric that is undefined everywhere.

The reference model of a normalized experiment is not among ``ExperimentResult.models``. The
leaderboard and the critical difference diagram rank it beside them from ``reference_runs``, so the
``*_RANKED_MODELS`` requirements count it; ``MULTIPLE_MODELS`` counts only the trained models, which
is what a model-against-model plot such as ``comparison_scatter`` compares.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult


class PlotRequirement(Enum):
    """Requirements that a plot class may declare."""

    #: Two or more trained models, without the reference model of a normalized experiment.
    MULTIPLE_MODELS = auto()
    #: Two or more models, the reference model of a normalized experiment counted.
    MULTIPLE_RANKED_MODELS = auto()
    #: Three or more models, the reference model counted: the least the Friedman test of a critical
    #: difference diagram accepts.
    AT_LEAST_THREE_RANKED_MODELS = auto()
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
    n_ranked = experiment.n_models + (1 if experiment.reference_runs else 0)
    available = {
        PlotRequirement.MULTIPLE_MODELS: experiment.n_models >= 2,
        PlotRequirement.MULTIPLE_RANKED_MODELS: n_ranked >= 2,
        PlotRequirement.AT_LEAST_THREE_RANKED_MODELS: n_ranked >= 3,
        PlotRequirement.MULTIPLE_FOLDS: experiment.max_folds >= 2,
        PlotRequirement.RANDOMIZATION: experiment.has_randomization,
        PlotRequirement.ROBUSTNESS: experiment.has_robustness,
    }
    return all(available[requirement] for requirement in requirements)
