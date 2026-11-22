"""Per-group Pearson correlations of every model, for the comparison plots.

The comparison scatter needs one number per (model, group) pair - the Pearson correlation of
a model's predictions against ground truth restricted to a single drug, sample or study - not
the underlying point cloud. :func:`model_group_correlations` concatenates the non-randomized
runs of each model, scores them on one shared, sorted group axis with
:func:`~drevalpy.evaluation.evaluate_per_group` and keeps a ``models x groups`` ``float32``
:class:`~pandas.DataFrame` whose index is the model names and whose columns are the group names.

The undefined-score rule is the one shared with every other per-group score (see
:mod:`drevalpy.evaluation._metrics`): a group with fewer than two pairs or a constant response is
``NaN`` for every model, and a constant predictor scores ``0.0`` within a group and is drawn.
A model whose correlations are all ``NaN`` is left out of the matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._grouping import defined_labels, group_labeller
from ._metrics import Metric
from ._scoring import evaluate_per_group

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from drevalpy.types.results import ExperimentResult, RunResult


def _shared_group_axis(result: ExperimentResult, labels_of: Callable[[RunResult], np.ndarray]) -> np.ndarray:
    """Return the sorted union of the labels ``labels_of`` gives over every scoring run."""
    seen: list[np.ndarray] = []
    for model in result.models:
        for run in model.select_runs():
            labels = labels_of(run)
            seen.append(np.unique(labels[defined_labels(labels)]))
    if not seen:
        return np.empty(0, dtype=object)
    return np.unique(np.concatenate(seen))


def model_group_correlations(result: ExperimentResult, grouping: str) -> pd.DataFrame:
    """Correlate predictions against ground truth per model and per group.

    The non-randomized runs of each model are concatenated and scored once on the shared
    group axis, so a group's value pools all folds of that model. A model with no defined
    correlation in any group (no group can support one) is left out; a constant predictor
    scores ``0.0`` within a group and is kept.

    :param result: Experiment whose models to summarise. Randomized runs are
        skipped.
    :param grouping: One of :data:`~drevalpy.evaluation._grouping.GROUPINGS`.
    :returns: A ``float32`` frame of Pearson correlations, one row per model of ``result`` with a defined
        correlation (index: model names) and one column per group (columns: group names, as strings).
    """
    import pandas as pd

    labels_of = group_labeller(grouping)
    groups = _shared_group_axis(result, labels_of)

    model_names: list[str] = []
    rows: list[np.ndarray] = []
    for model in result.models:
        runs = model.select_runs()
        if not runs:
            continue
        scores = evaluate_per_group(
            np.concatenate([run.predictions for run in runs]),
            np.concatenate([run.ground_truth for run in runs]),
            Metric.PEARSON,
            labels=np.concatenate([labels_of(run) for run in runs]),
            groups=groups,
        )
        row = scores.values[Metric.PEARSON].astype(np.float32)
        if np.isfinite(row).any():
            rows.append(row)
            model_names.append(model.model_name)

    values = np.stack(rows) if rows else np.empty((0, len(groups)), dtype=np.float32)
    return pd.DataFrame(values, index=model_names, columns=[str(name) for name in groups])
