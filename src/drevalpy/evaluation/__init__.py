"""Evaluation metrics, pooled over all pairs or computed within each drug, sample, study or annotation group.

:class:`Metric` lists the metrics. Each member is its stored name (``Metric.PEARSON == "Pearson"``),
``Metric(name)`` looks a name up, ``.direction`` says which way is better and ``.kind`` what it
measures. :func:`evaluate` computes pooled scores and :func:`evaluate_per_group` per-group scores of
arrays. For an experiment, :meth:`drevalpy.types.results.ExperimentResult.summary_table` with
``group_by`` gives the per-group scores of every model.
"""

from ._grouping import GroupBy
from ._metrics import Metric, MetricKind
from ._scoring import PerGroupScores, evaluate, evaluate_per_group

__all__ = [
    "GroupBy",
    "Metric",
    "MetricKind",
    "PerGroupScores",
    "evaluate",
    "evaluate_per_group",
]
