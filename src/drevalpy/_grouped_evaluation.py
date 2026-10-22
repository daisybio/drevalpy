"""Per-group metrics: one score per drug, sample or study, and how to average them.

Pooled metrics reward a model for knowing which drugs are potent. A per-group score
removes that: the metric is computed among the pairs of a single drug (or sample), so it
asks whether the model ranks the *other* axis correctly. The ``"study"`` grouping reads
the study from the ``"<study>::"`` prefix of the sample IDs that ``drevalpy.data.concat``
writes, so it scores each study of a concatenated dataset separately. This module is the one
place that rule lives; :meth:`drevalpy.types.results.ExperimentResult.summary_table` and the
comparison plots in :mod:`drevalpy.visualization.plots._group_metrics` both call it.

**Undefined groups.** A group's score is ``NaN`` - undefined, never ``0.0`` and never
``1.0`` - when:

* it does not exist: a pair whose sample ID has no ``"::"`` prefix belongs to no study, so
  on a run of a single, non-concatenated dataset the ``"study"`` grouping has no groups, or
* it has fewer than two usable pairs (correlations and ``R^2``), or
* a correlation (``Pearson``, ``Spearman``, ``Kendall``) has predictions or response with
  no variance, i.e. a range of at most :data:`CONSTANT_TOLERANCE`, or
* ``R^2`` has a response with no variance.

A model that predicts a constant within each group - the per-drug mean baseline, for
instance - therefore has no per-drug correlation at all, rather than a score that could
be mistaken for a good or a neutral one. Error metrics (``MSE``, ``RMSE``, ``MAE``) are
defined for any group with one pair. Pairs where either value is not finite are dropped
before anything is counted.

**Averaging.** Undefined groups are skipped, not counted as zero, so a mean over groups
is ``NaN`` only when no group is defined; :func:`mean_over_groups` returns how many were.

``scipy`` and ``sklearn`` are only reached through :data:`drevalpy.evaluation.AVAILABLE_METRICS`,
which imports them lazily, so importing this module costs nothing beyond ``numpy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from drevalpy._grouped_pearson import CONSTANT_TOLERANCE, MIN_GROUP_SIZE, PearsonSums

if TYPE_CHECKING:
    from drevalpy.types.results import RunResult

#: What a result can be grouped by, in report order.
GROUPINGS: Final[tuple[str, ...]] = ("drug", "sample", "study")

#: A grouping of pairs: by drug, by sample or by study.
GroupBy = Literal["drug", "sample", "study"]

#: Grouping name -> the ``RunResult`` attribute holding that grouping's labels.
_ID_ATTRIBUTE: Final[dict[str, str]] = {"drug": "drug_ids", "sample": "sample_ids", "study": "sample_ids"}

#: Separates the study from the source sample ID in the sample IDs of a concatenated dataset.
STUDY_SEPARATOR: Final[str] = "::"

_CORRELATIONS: Final[frozenset[str]] = frozenset({"Pearson", "Spearman", "Kendall"})
_NEEDS_VARYING_RESPONSE: Final[frozenset[str]] = _CORRELATIONS | {"R^2"}


def group_labels(run: RunResult, grouping: str) -> np.ndarray:
    """Return the grouping labels of a run.

    :param run: Run whose identifier arrays to read.
    :param grouping: One of :data:`GROUPINGS`.
    :returns: The run's ``drug_ids`` or ``sample_ids`` array. For ``"study"``, an object array
        of the sample ID prefixes before ``"::"`` (the study of a concatenated dataset), NaN
        where a sample ID has no prefix; see :func:`defined_labels`.
    :raises ValueError: If ``grouping`` is not one of :data:`GROUPINGS`.
    """
    try:
        attribute = _ID_ATTRIBUTE[grouping]
    except KeyError:
        raise ValueError(f"Unknown grouping {grouping!r}; expected one of {GROUPINGS}") from None
    labels = np.asarray(getattr(run, attribute))
    return _study_of(labels) if grouping == "study" else labels


def _study_of(sample_ids: np.ndarray) -> np.ndarray:
    """Return the prefix before ``"::"`` of every sample ID, NaN where there is none."""
    distinct, inverse = np.unique(sample_ids.astype(str), return_inverse=True)
    prefixes = np.empty(len(distinct), dtype=object)
    for position, sample_id in enumerate(distinct):
        study, separator, _ = sample_id.partition(STUDY_SEPARATOR)
        prefixes[position] = study if separator and study else np.nan
    return prefixes[inverse]


def defined_labels(labels: np.ndarray) -> np.ndarray:
    """Return which pairs have a group, i.e. a label that is not NaN.

    :param labels: Labels as returned by :func:`group_labels`.
    :returns: Boolean array, all True unless the labels are an object array holding NaN.
    """
    if labels.dtype != object:
        return np.ones(labels.shape, dtype=bool)
    return np.fromiter((isinstance(label, str) for label in labels), dtype=bool, count=len(labels))


def _group_score(name: str, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Score one group with the registered metric, returning NaN where it is undefined.

    :param name: Key of :data:`drevalpy.evaluation.AVAILABLE_METRICS`.
    :param y_pred: Predictions of the group's pairs.
    :param y_true: Observed response of the same pairs.
    :returns: The score, or NaN under the rule in the module docstring.
    """
    from drevalpy.evaluation import AVAILABLE_METRICS

    if name in _NEEDS_VARYING_RESPONSE and (len(y_true) < MIN_GROUP_SIZE or np.ptp(y_true) <= CONSTANT_TOLERANCE):
        return float("nan")
    if name in _CORRELATIONS and np.ptp(y_pred) <= CONSTANT_TOLERANCE:
        return float("nan")
    return float(AVAILABLE_METRICS[name](y_pred=y_pred, y_true=y_true))


def _within_group_ranks(codes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Rank ``values`` within each group, ties sharing their average rank.

    :param codes: Group index per observation, ``0 .. n_groups - 1``.
    :param values: Finite values, aligned with ``codes``.
    :returns: Ranks starting at 1 within every group.
    """
    from scipy.stats import rankdata

    dense = rankdata(values, method="dense")
    key = codes.astype(np.int64) * (int(dense.max()) + 1) + dense.astype(np.int64)
    first_rank = np.concatenate(([0], np.cumsum(np.bincount(codes))))[codes]
    return rankdata(key, method="average") - first_rank


def _vectorised_scores(
    name: str, codes: np.ndarray, n_groups: int, y_pred: np.ndarray, y_true: np.ndarray, sums: PearsonSums
) -> np.ndarray | None:
    """Compute a metric for every group at once, if a bincount formulation exists.

    These agree with :data:`drevalpy.evaluation.AVAILABLE_METRICS` applied group by group
    (pinned by ``tests/test_grouped_evaluation.py``); they exist because that loop costs
    seconds per run at hundreds of groups.

    :param name: Metric name.
    :param codes: Group index per usable pair.
    :param n_groups: Number of groups.
    :param y_pred: Predictions aligned with ``codes``.
    :param y_true: Response aligned with ``codes``.
    :param sums: Sums already accumulated from ``y_pred`` and ``y_true``.
    :returns: One score per group, or ``None`` when ``name`` has no vectorised form.
    """
    counts = np.maximum(sums.counts, 1)
    error = y_pred - y_true
    if name == "Pearson":
        return sums.correlations()
    if name == "Spearman":
        ranks = PearsonSums(n_groups)
        ranks.add(codes, _within_group_ranks(codes, y_pred), _within_group_ranks(codes, y_true))
        return np.where(sums.varies(), ranks.correlations(), np.nan)
    if name in {"MSE", "RMSE"}:
        mse = np.bincount(codes, weights=error * error, minlength=n_groups) / counts
        return np.sqrt(mse) if name == "RMSE" else mse
    if name == "MAE":
        return np.bincount(codes, weights=np.abs(error), minlength=n_groups) / counts
    if name == "R^2":
        total = sums.sum_squares_of_y()
        defined = (sums.counts >= MIN_GROUP_SIZE) & sums.y_varies() & (total > 0)
        sse = np.bincount(codes, weights=error * error, minlength=n_groups)
        return np.where(defined, 1.0 - sse / np.where(defined, total, 1.0), np.nan)
    return None


def grouped_scores(
    labels: np.ndarray,
    predictions: np.ndarray,
    response: np.ndarray,
    metrics: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute metrics within each group of pairs.

    :param labels: Group label per pair, for example the drug ids of a run. Pairs whose label
        is NaN (see :func:`defined_labels`) belong to no group and are dropped.
    :param predictions: Predicted response per pair.
    :param response: Observed response per pair.
    :param metrics: Metric names from :data:`drevalpy.evaluation.AVAILABLE_METRICS`; ``None``
        selects all of them.
    :returns: The sorted distinct labels that have at least one usable pair, and for each
        metric a ``float64`` array of one score per label, NaN where undefined.
    :raises ValueError: If the inputs differ in length or a metric name is unknown.
    """
    from drevalpy.evaluation import AVAILABLE_METRICS

    labels, predictions, response = np.asarray(labels), np.asarray(predictions), np.asarray(response)
    if not labels.shape == predictions.shape == response.shape or labels.ndim != 1:
        raise ValueError("labels, predictions and response must be one-dimensional and equally long")
    names = list(AVAILABLE_METRICS) if metrics is None else metrics
    for name in names:
        if name not in AVAILABLE_METRICS:
            raise ValueError(f"invalid metric {name}. Available: {list(AVAILABLE_METRICS)}")

    usable = np.isfinite(predictions.astype(np.float64)) & np.isfinite(response.astype(np.float64))
    usable &= defined_labels(labels)
    groups, codes = np.unique(labels[usable], return_inverse=True)
    y_pred, y_true = predictions[usable].astype(np.float64), response[usable].astype(np.float64)

    sums = PearsonSums(len(groups))
    sums.add(codes, y_pred, y_true)
    order = bounds = None
    scores: dict[str, np.ndarray] = {}
    for name in names:
        vectorised = _vectorised_scores(name, codes, len(groups), y_pred, y_true, sums)
        if vectorised is not None:
            scores[name] = np.asarray(vectorised, dtype=np.float64)
            continue
        if order is None:
            order = np.argsort(codes, kind="stable")
            bounds = np.searchsorted(codes[order], np.arange(len(groups) + 1))
        scores[name] = np.array(
            [
                _group_score(name, y_pred[order[lo:hi]], y_true[order[lo:hi]])
                for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
            ],
            dtype=np.float64,
        )
    return groups, scores


def mean_over_groups(values: np.ndarray) -> tuple[float, int]:
    """Average per-group scores, skipping the undefined ones.

    :param values: One score per group, NaN where undefined.
    :returns: The mean over the defined groups (NaN if there are none) and how many
        groups were defined.
    """
    defined = values[np.isfinite(values)]
    return (float(defined.mean()) if defined.size else float("nan")), int(defined.size)
