"""Bounded, vectorised per-group correlation metrics for the comparison plots.

The comparison scatter needs one number per (model, group) pair - the Pearson
correlation of a model's predictions against ground truth restricted to a single
drug or sample - not the underlying point cloud. Computing it with
``DataFrame.groupby(...).apply(pearsonr)`` costs a Python call per group and a
row-wise DataFrame to group over; at 96 models x 10 folds x 23k rows that
dominates both runtime and peak memory.

Everything here works on ``np.bincount`` sums instead, so the cost is a handful
of vectorised passes per fold and the retained result is a ``models x groups``
float32 matrix - 0.2 MB at the scale above.

The per-group correlation itself, and the rule for groups where it is undefined, live in
:mod:`drevalpy._grouped_evaluation` and :mod:`drevalpy._grouped_pearson`, shared with
``ExperimentResult.summary_table``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from drevalpy._grouped_evaluation import GROUPINGS, group_labels
from drevalpy._grouped_pearson import MIN_GROUP_SIZE, PearsonSums

__all__ = [
    "GROUPINGS",
    "GROUPING_LABELS",
    "MIN_GROUP_SIZE",
    "GroupCorrelationMatrix",
    "group_labels",
    "model_group_correlations",
]

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult, ModelResult, RunResult

#: Human-readable axis/section labels per grouping.
GROUPING_LABELS: Final[dict[str, str]] = {"drug": "drug", "sample": "sample"}


def _scoring_runs(model: ModelResult) -> list[RunResult]:
    """Return the non-randomized runs of a model."""
    return model.select_runs()


@dataclass(frozen=True)
class GroupCorrelationMatrix:
    """Per-group correlations for every model, on a shared group axis.

    The shared axis is what makes the comparison plot cheap: two models are
    compared by reading two rows of :attr:`values`, with column ``j`` of both
    referring to ``group_names[j]``.
    """

    grouping: str
    model_names: tuple[str, ...]
    group_names: tuple[str, ...]
    values: np.ndarray

    @property
    def n_models(self) -> int:
        """Number of models on the row axis."""
        return len(self.model_names)

    @property
    def n_groups(self) -> int:
        """Number of drugs or samples on the column axis."""
        return len(self.group_names)

    @property
    def is_empty(self) -> bool:
        """Whether the matrix has no models or no groups."""
        return self.n_models == 0 or self.n_groups == 0

    def for_model(self, model_name: str) -> np.ndarray:
        """Return one model's correlation vector.

        :param model_name: Name of a model in :attr:`model_names`.
        :returns: A read-only view of the matching row of :attr:`values`.
        :raises KeyError: If ``model_name`` is not in :attr:`model_names`.
        """
        try:
            index = self.model_names.index(model_name)
        except ValueError:
            raise KeyError(model_name) from None
        return self.values[index]

    def drop_all_nan_models(self) -> GroupCorrelationMatrix:
        """Return a copy without models whose correlations are all NaN.

        A model that predicts a constant within every group - the per-drug and
        per-sample naive baselines do exactly that - has no defined
        correlation anywhere and would only contribute an empty dropdown entry.

        :returns: ``self`` when nothing is dropped, otherwise a filtered copy.
        """
        if self.is_empty:
            return self
        keep = np.isfinite(self.values).any(axis=1)
        if keep.all():
            return self
        return GroupCorrelationMatrix(
            grouping=self.grouping,
            model_names=tuple(np.asarray(self.model_names)[keep].tolist()),
            group_names=self.group_names,
            values=self.values[keep],
        )


def _shared_group_axis(result: ExperimentResult, grouping: str) -> np.ndarray:
    """Return the sorted union of group labels over every scoring run."""
    seen: list[np.ndarray] = []
    for model in result.models:
        for run in _scoring_runs(model):
            seen.append(np.unique(group_labels(run, grouping)))
    if not seen:
        return np.empty(0, dtype=object)
    return np.unique(np.concatenate(seen))


def model_group_correlations(
    result: ExperimentResult,
    grouping: str,
    *,
    min_count: int = MIN_GROUP_SIZE,
) -> GroupCorrelationMatrix:
    """Correlate predictions against ground truth per model and per group.

    Folds are pooled per model and consumed one at a time, so the retained
    memory is the ``n_models x n_groups`` result rather than anything
    proportional to the number of predictions.

    :param result: Experiment whose models to summarise. Randomized runs are
        skipped.
    :param grouping: One of :data:`GROUPINGS`.
    :param min_count: Minimum observations a group needs for a finite value.
    :returns: A :class:`GroupCorrelationMatrix` over every model in ``result``.
    """
    groups = _shared_group_axis(result, grouping)
    import pandas as pd

    group_index = pd.Index(groups)
    n_groups = len(groups)

    model_names: list[str] = []
    rows: list[np.ndarray] = []
    for model in result.models:
        runs = _scoring_runs(model)
        if not runs:
            continue
        sums = PearsonSums(n_groups)
        for run in runs:
            codes = group_index.get_indexer(group_labels(run, grouping))
            sums.add(codes, run.predictions, run.ground_truth)
        model_names.append(model.model_name)
        rows.append(sums.correlations(min_count=min_count).astype(np.float32))

    values = np.stack(rows) if rows else np.empty((0, n_groups), dtype=np.float32)
    return GroupCorrelationMatrix(
        grouping=grouping,
        model_names=tuple(model_names),
        group_names=tuple(str(name) for name in groups),
        values=values,
    )
