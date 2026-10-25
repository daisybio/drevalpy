"""Per-group Pearson correlations of every model, for the comparison plots.

The comparison scatter needs one number per (model, group) pair - the Pearson correlation of
a model's predictions against ground truth restricted to a single drug, sample or study - not
the underlying point cloud. :func:`model_group_correlations` concatenates the non-randomized
runs of each model, scores them on one shared, sorted group axis with
:func:`~drevalpy.evaluation.evaluate_per_group` and keeps a ``models x groups`` ``float32`` matrix.

The undefined-score rule is the one shared with every other per-group score (see
:mod:`drevalpy.evaluation._metrics`): a group with fewer than two pairs or a constant response is
``NaN`` for every model, and a constant predictor scores ``0.0`` within a group and is drawn.
Groups with fewer than ``min_count`` pairs are additionally masked to ``NaN``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._grouping import defined_labels, group_labels
from ._metrics import MIN_GROUP_SIZE, Metric
from ._scoring import evaluate_per_group

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult, ModelResult, RunResult


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
        """Number of drugs, samples or studies on the column axis."""
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

        A model has no defined correlation anywhere only when no group can support one
        (for example a constant response everywhere or every group below ``min_count``);
        a constant predictor scores ``0.0`` within a group and is kept.

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
            labels = group_labels(run, grouping)
            seen.append(np.unique(labels[defined_labels(labels)]))
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

    The non-randomized runs of each model are concatenated and scored once on the shared
    group axis, so a group's value pools all folds of that model.

    :param result: Experiment whose models to summarise. Randomized runs are
        skipped.
    :param grouping: One of :data:`GROUPINGS`.
    :param min_count: Minimum observations a group needs for a finite value.
    :returns: A :class:`GroupCorrelationMatrix` over every model in ``result``.
    """
    groups = _shared_group_axis(result, grouping)

    model_names: list[str] = []
    rows: list[np.ndarray] = []
    for model in result.models:
        runs = _scoring_runs(model)
        if not runs:
            continue
        scores = evaluate_per_group(
            np.concatenate([run.predictions for run in runs]),
            np.concatenate([run.ground_truth for run in runs]),
            Metric.PEARSON,
            labels=np.concatenate([group_labels(run, grouping) for run in runs]),
            groups=groups,
        )
        row = np.where(scores.n_pairs < min_count, np.nan, scores.values[Metric.PEARSON])
        rows.append(row.astype(np.float32))
        model_names.append(model.model_name)

    values = np.stack(rows) if rows else np.empty((0, len(groups)), dtype=np.float32)
    return GroupCorrelationMatrix(
        grouping=grouping,
        model_names=tuple(model_names),
        group_names=tuple(str(name) for name in groups),
        values=values,
    )
