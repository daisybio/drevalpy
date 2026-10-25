"""Applying the metrics: pooled scores over all pairs, and per-group scores within each group of pairs.

Pooled metrics reward a model for knowing which drugs are potent. A per-group score removes that:
the metric is computed among the pairs of a single drug (or sample), so it asks whether the model
ranks the *other* axis correctly. A grouping that pools many drugs (a tissue, say) still rewards
knowing drug potency. Groupings are defined in :mod:`drevalpy.evaluation._grouping`.

Pooled and per-group scores both end in :func:`~drevalpy.evaluation._metrics.score_pairs`, so they
share the undefined-score rule. A model that predicts a constant within each drug - the per-drug
mean baseline, for instance - therefore scores ``0.0`` within every drug for a correlation.

**Averaging.** Undefined groups are skipped, so a mean over groups is ``NaN`` only when no group is
defined; :meth:`PerGroupScores.n_defined` says how many were. Which groups are undefined depends
only on the data, so every model scored on the same pairs is averaged over the same groups.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._grouping import GroupBy, defined_labels, group_labels
from ._metrics import Metric, score_pairs
from ._pairs import Pairs, group_pairs

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from drevalpy.types.results import RunResult

#: Which metrics to score: ``None`` for all of them, one name, or several names. Members of
#: :class:`~drevalpy.evaluation.Metric` are names, so they are accepted wherever a name is.
MetricSelection = Metric | str | Iterable[Metric | str] | None


def _select(metrics: MetricSelection) -> tuple[Metric, ...]:
    """Look up the selected metrics, rejecting unknown names before any work.

    :param metrics: ``None``, one name, or several names.
    :returns: The metrics in the requested order; every metric in report order for ``None``.
    """
    if metrics is None:
        return tuple(Metric)
    if isinstance(metrics, str):
        return (Metric(metrics),)
    return tuple(Metric(name) for name in metrics)


def mean_of_defined(values: ArrayLike) -> float:
    """Average scores, skipping the undefined (non-finite) ones.

    :param values: Scores, ``NaN`` where undefined.
    :returns: The mean of the defined scores, ``NaN`` if there are none.
    """
    values = np.asarray(values, dtype=np.float64)
    defined = values[np.isfinite(values)]
    return float(defined.mean()) if defined.size else float("nan")


def evaluate(predictions: ArrayLike, response: ArrayLike, metrics: MetricSelection = None) -> dict[str, float]:
    """Compute pooled scores: each metric over all pairs where prediction and response are finite.

    Raises ``ValueError`` if the inputs are not one-dimensional and equally long, or a name is unknown.

    :param predictions: Prediction per pair.
    :param response: Observed response per pair, in the same order.
    :param metrics: The metrics to score: ``None`` for all of them, one name, or several names.
    :returns: Score by metric name, in the requested order; ``NaN`` where the data cannot support a
        metric and ``0.0`` for a correlation of a constant prediction (see :mod:`drevalpy.evaluation._metrics`).
    """
    selected = _select(metrics)
    pairs = Pairs.finite(predictions, response)
    return {metric.value: score_pairs(metric, pairs) for metric in selected}


def run_metrics(predictions: ArrayLike, response: ArrayLike) -> dict[str, float]:
    """Compute the pooled scores a run or a tuning trial records in its ``metrics``.

    Every metric, as :func:`evaluate` scores it, except that a run without a single usable pair records
    ``{}``: it was not scored at all, so fold averages and tuning skip it instead of turning ``NaN``.

    Raises ``ValueError`` if the inputs are not one-dimensional and equally long.

    :param predictions: Prediction per pair of the run.
    :param response: Observed response per pair.
    :returns: Score by metric name for every metric, or ``{}``.
    """
    pairs = Pairs.finite(predictions, response)
    return {metric.value: score_pairs(metric, pairs) for metric in Metric} if len(pairs) else {}


@dataclass(frozen=True, eq=False)
class PerGroupScores:
    """Scores of several metrics within each group of pairs, on one group axis.

    Position ``j`` of every array refers to ``groups[j]``. :meth:`mean` gives the per-group score of
    each metric; :attr:`values` holds the score within each group.
    """

    #: The group axis: the sorted labels of the groups.
    groups: np.ndarray
    #: Number of usable pairs in each group; ``0`` for a group of a fixed axis without pairs.
    n_pairs: np.ndarray
    #: Metric name -> one ``float64`` score per group, ``NaN`` where undefined.
    values: Mapping[str, np.ndarray]

    def mean(self) -> dict[str, float]:
        """Average each metric over the groups where it is defined.

        :returns: Mean by metric name, ``NaN`` for a metric no group could score.
        """
        return {name: mean_of_defined(scores) for name, scores in self.values.items()}

    def n_defined(self) -> dict[str, int]:
        """Count the groups behind each :meth:`mean`.

        :returns: Number of groups with a defined score, by metric name.
        """
        return {name: int(np.isfinite(scores).sum()) for name, scores in self.values.items()}


def evaluate_per_group(
    predictions: ArrayLike,
    response: ArrayLike,
    metrics: MetricSelection = None,
    *,
    labels: ArrayLike,
    groups: ArrayLike | None = None,
) -> PerGroupScores:
    """Compute each metric within every group of pairs.

    An unknown metric name also raises ``ValueError``.

    :param predictions: Prediction per pair.
    :param response: Observed response per pair, in the same order.
    :param metrics: The metrics to score, as for :func:`evaluate`.
    :param labels: Group label per pair, such as the drug of each pair; a NaN label puts the pair in no group.
    :param groups: A fixed, sorted group axis shared by several calls (for example one per model);
        groups without pairs are kept and score ``NaN``. ``None`` uses the sorted labels of the usable pairs.
    :returns: The scores within each group and their means.
    :raises ValueError: If the inputs are not one-dimensional and equally long.
    """
    selected = _select(metrics)
    labels, predictions, response = np.asarray(labels), np.asarray(predictions), np.asarray(response)
    if labels.ndim != 1 or not labels.shape == predictions.shape == response.shape:
        raise ValueError("Predictions, response and labels must be one-dimensional and equally long")
    grouped = defined_labels(labels)
    axis, per_group = group_pairs(
        labels[grouped],
        predictions[grouped],
        response[grouped],
        None if groups is None else np.asarray(groups),
    )
    values = {
        metric.value: np.array([score_pairs(metric, pairs) for pairs in per_group], dtype=np.float64)
        for metric in selected
    }
    return PerGroupScores(axis, np.array([len(pairs) for pairs in per_group], dtype=np.int64), values)


def run_per_group_score(run: RunResult, group_by: GroupBy, metric: Metric | str) -> float:
    """Compute one metric's per-group score of a run: its mean over the groups where it is defined.

    :param run: Run whose predictions and response to score.
    :param group_by: One of :data:`~drevalpy.evaluation._grouping.GROUPINGS`.
    :param metric: The metric.
    :returns: The mean over the defined groups, ``NaN`` if there are none.
    """
    metric = Metric(metric)
    scores = evaluate_per_group(run.predictions, run.ground_truth, metric, labels=group_labels(run, group_by))
    return scores.mean()[metric.value]
