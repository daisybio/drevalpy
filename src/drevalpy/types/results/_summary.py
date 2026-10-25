"""Summary tables over the models of an experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from drevalpy.evaluation._grouping import GroupBy
    from drevalpy.types.data.dataset import Dataset
    from drevalpy.types.results.model import ModelResult


class SummaryTableMixin:
    """The ``summary_table`` views of :class:`~drevalpy.types.results.ExperimentResult`."""

    models: list[ModelResult]

    def summary_table(self, *, group_by: GroupBy | None = None, dataset: Dataset | None = None) -> pd.DataFrame:
        """Mean metric per model, one row per model and one column per metric.

        Without ``group_by`` each metric is the pooled score over all pairs of a fold, averaged over folds. With
        ``group_by`` it is the per-group score within each group of a fold, averaged over the groups where it is
        defined, then averaged over folds; that tells a model that only knows which drugs are potent (high pooled
        score, about zero per drug) from one that ranks samples within a drug. :doc:`/reference/metrics` defines the
        groups and when a score is undefined (``NaN``). Use :meth:`summary_group_counts` for how many groups were
        defined. Randomized and robustness runs are not part of either table.

        Raises ``ValueError`` if ``group_by`` is neither built in nor an unambiguous annotation column of
        ``dataset``, or needs a ``dataset`` that was not given.

        :param group_by: ``None`` for pooled metrics, or ``"drug"`` / ``"sample"`` / ``"study"`` / a sample or
            drug annotation column of ``dataset``.
        :param dataset: Dataset the runs were made on; needed only for an annotation grouping.
        :returns: Means over folds, indexed by model name.
        """
        return self._summary("mean", group_by, dataset)

    def summary_group_counts(self, group_by: GroupBy, dataset: Dataset | None = None) -> pd.DataFrame:
        """Count the groups with a defined score behind ``summary_table(group_by=..., dataset=...)``.

        Raises ``ValueError`` as :meth:`summary_table` does.

        :param group_by: ``"drug"``, ``"sample"``, ``"study"`` or an annotation column of ``dataset``.
        :param dataset: Dataset the runs were made on; needed only for an annotation grouping.
        :returns: The mean over folds of the number of groups where the metric is defined, indexed by model name
            like :meth:`summary_table`. The counts are the same for every model scored on the same pairs.
        """
        return self._summary("n_groups", group_by, dataset)

    def _summary(self, statistic: str, group_by: GroupBy | None, dataset: Dataset | None) -> pd.DataFrame:
        """Build the model-by-metric table of one statistic.

        :param statistic: ``"mean"``, or ``"n_groups"`` for a grouped summary.
        :param group_by: ``None`` for pooled metrics, else the grouping of ``aggregate_group_metrics``.
        :param dataset: Dataset an annotation grouping reads.
        :returns: The table, indexed by model name; a model missing a metric gets NaN.
        """
        import pandas as pd

        rows = {}
        for model in self.models:
            aggregate = (
                model.aggregate_metrics if group_by is None else model.aggregate_group_metrics(group_by, dataset)
            )
            rows[model.model_name] = {metric: stats[statistic] for metric, stats in aggregate.items()}
        return pd.DataFrame.from_dict(rows, orient="index", dtype=float)
