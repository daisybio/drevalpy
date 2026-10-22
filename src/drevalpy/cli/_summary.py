"""The ``--group-by`` option and the grouped summary it prints, shared by ``run`` and ``aggregate``."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult


class GroupBy(StrEnum):
    """Values of ``--group-by``: compute each metric within every drug or sample."""

    DRUG = "drug"
    SAMPLE = "sample"


def format_grouped_summary(experiment: ExperimentResult, group_by: GroupBy) -> str:
    """Render ``summary_table(group_by=...)`` as aligned text.

    Each cell is the mean over groups and folds followed by the mean number of groups it is
    based on; ``undefined`` marks a metric no group could score, such as the Pearson
    correlation of a constant prediction.

    :param experiment: Experiment to summarize.
    :param group_by: Whether to group by drug or by sample.
    :returns: A heading and one row per model.
    """
    grouping = group_by.value
    means = experiment.summary_table(group_by=grouping)
    counts = experiment.summary_group_counts(group_by=grouping)
    metrics = list(dict.fromkeys(metric for row in means.values() for metric in row))
    table = [["Model", *metrics]]
    for name, row in means.items():
        table.append([name, *(_cell(row.get(metric, math.nan), counts[name].get(metric, 0.0)) for metric in metrics)])
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in table]
    heading = (
        f"Metrics computed within each {grouping}, averaged over {grouping}s and folds "
        f"(number of {grouping}s scored in brackets):"
    )
    return "\n".join([heading, *lines])


def _cell(mean: float, groups: float) -> str:
    """Format one table cell.

    :param mean: Mean over groups and folds, NaN if no group was defined.
    :param groups: Mean number of groups with a defined score per fold.
    :returns: ``"<mean> (<groups>)"``, or ``"undefined"``.
    """
    return "undefined" if math.isnan(mean) else f"{mean:.3f} ({groups:g})"
