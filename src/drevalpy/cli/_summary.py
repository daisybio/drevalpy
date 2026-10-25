"""The ``--group-by`` option and the grouped summary it prints, shared by ``run`` and ``aggregate``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import typer

from drevalpy.evaluation._grouping import GROUPINGS, group_labeller

if TYPE_CHECKING:
    from drevalpy.types import Dataset
    from drevalpy.types.results import ExperimentResult


def normalize_group_by(value: str | None) -> str | None:
    """Lowercase the built-in groupings of ``--group-by``; annotation columns are case-sensitive.

    :param value: The option value.
    :returns: ``"drug"``, ``"sample"`` or ``"study"`` for any capitalization of them, else *value*.
    """
    return value.lower() if value is not None and value.lower() in GROUPINGS else value


def resolve_group_by_dataset(group_by: str | None, dataset: list[str] | None) -> Dataset | None:
    """Load the dataset an annotation ``--group-by`` reads, and check the grouping against it.

    Nothing is loaded for a built-in grouping, which needs no dataset.

    :param group_by: The ``--group-by`` value.
    :param dataset: The ``--dataset`` values, names or ``.h5mu`` paths.
    :returns: The (concatenated) dataset, or ``None`` if ``group_by`` is unset or built in.
    :raises typer.BadParameter: If an annotation grouping has no dataset, or names no unambiguous annotation column.
    """
    if group_by is None or group_by in GROUPINGS:
        return None
    if not dataset:
        raise typer.BadParameter(
            f"--group-by {group_by} reads a dataset annotation; pass --dataset (or use one of {', '.join(GROUPINGS)}).",
            param_hint="--group-by",
        )
    from drevalpy._run import _resolve_dataset

    resolved = _resolve_dataset(dataset)
    try:
        group_labeller(group_by, resolved)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--group-by") from error
    return resolved


def format_grouped_summary(experiment: ExperimentResult, group_by: str, dataset: Dataset | None = None) -> str:
    """Render ``summary_table(group_by=...)`` as aligned text.

    Each cell is the mean over groups and folds followed by the mean number of groups it is
    based on; ``undefined`` marks a metric no group could score, such as a correlation
    when every drug has a constant response.

    :param experiment: Experiment to summarize.
    :param group_by: Drug, sample, study, or a sample or drug annotation column of ``dataset``.
    :param dataset: Dataset the annotation is read from; needed only for an annotation grouping.
    :returns: A heading and one row per model.
    """
    means = experiment.summary_table(group_by=group_by, dataset=dataset)
    counts = experiment.summary_group_counts(group_by, dataset)
    metrics = list(means.columns)
    table = [["Model", *metrics]]
    for name, row in means.iterrows():
        count = counts.loc[name] if name in counts.index else {}
        table.append([str(name), *(_cell(row[metric], count.get(metric, 0.0)) for metric in metrics)])
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in table]
    if group_by in GROUPINGS:
        unit, plural = group_by, "studies" if group_by == "study" else f"{group_by}s"
    else:
        prefix, _, column = group_by.partition(":")
        unit, plural = f"{column if prefix in ('sample', 'drug') else group_by} group", "groups"
    heading = (
        f"Metrics computed within each {unit}, averaged over {plural} and folds "
        f"(number of {plural} scored in brackets):"
    )
    return "\n".join([heading, *lines])


def _cell(mean: float, groups: float) -> str:
    """Format one table cell.

    :param mean: Mean over groups and folds, NaN if no group was defined.
    :param groups: Mean number of groups with a defined score per fold.
    :returns: ``"<mean> (<groups>)"``, or ``"undefined"``.
    """
    return "undefined" if math.isnan(mean) else f"{mean:.3f} ({groups:g})"
