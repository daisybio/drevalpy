"""Summary tables over the models of an experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drevalpy._grouped_evaluation import GroupBy
    from drevalpy.types.results.model import ModelResult


class SummaryTableMixin:
    """The ``summary_table`` views of :class:`~drevalpy.types.results.ExperimentResult`."""

    models: list[ModelResult]

    def summary_table(self, *, group_by: GroupBy | None = None) -> dict[str, dict[str, float]]:
        """Mean metric per model: {model_name: {metric: mean}}.

        Without ``group_by`` each metric is the pooled score over all pairs of a fold,
        averaged over folds. With ``group_by="drug"`` (or ``"sample"`` or ``"study"``) it is
        computed within each drug (sample, study) of a fold, averaged over the groups where it is defined, then
        averaged over folds. This is the number that tells a model that only knows which
        drugs are potent (high pooled, undefined per drug) from one that ranks samples within
        a drug. A group where the metric is undefined - fewer than two pairs, or no variance
        in the predictions or in the response, so a constant prediction has no per-drug
        Pearson - is skipped rather than scored 0, and the result is ``NaN`` if no group is
        defined. The rule is shared with the comparison plots, see
        :mod:`drevalpy._grouped_evaluation`. Use :meth:`summary_group_counts` for how many
        groups were defined. The study of a pair is the ``"<study>::"`` prefix of its sample ID, as written by
        ``drevalpy.data.concat``; a run without prefixes has no studies, so ``group_by="study"``
        is ``NaN``. Randomized and robustness runs are not part of either table.

        Raises ``ValueError`` if ``group_by`` is not ``None``, ``"drug"``, ``"sample"`` or ``"study"``.

        :param group_by: ``None`` for pooled metrics, or ``"drug"`` / ``"sample"`` / ``"study"``.
        :returns: Nested dict suitable for DataFrame construction.
        """
        if group_by is None:
            return {m.model_name: {k: v["mean"] for k, v in m.aggregate_metrics.items()} for m in self.models}
        return {
            m.model_name: {k: v["mean"] for k, v in m.aggregate_group_metrics(group_by).items()} for m in self.models
        }

    def summary_group_counts(self, group_by: GroupBy) -> dict[str, dict[str, float]]:
        """Count the drugs (samples, studies) with a defined score behind ``summary_table(group_by=...)``.

        Raises ``ValueError`` if ``group_by`` is not ``"drug"``, ``"sample"`` or ``"study"``.

        :param group_by: ``"drug"``, ``"sample"`` or ``"study"``.
        :returns: {model_name: {metric: groups}}, the mean over folds of the number of groups
            where the metric is defined. Error metrics count every group; correlations skip
            constant ones.
        """
        return {
            m.model_name: {k: v["n_groups"] for k, v in m.aggregate_group_metrics(group_by).items()}
            for m in self.models
        }
