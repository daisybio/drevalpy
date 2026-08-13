"""Bounded, vectorised per-group correlation metrics for the comparison plots.

The comparison scatter needs one number per (model, group) pair - the Pearson
correlation of a model's predictions against ground truth restricted to a single
drug or cell line - not the underlying point cloud. Computing it with
``DataFrame.groupby(...).apply(pearsonr)`` costs a Python call per group and a
row-wise DataFrame to group over; at 96 models x 10 folds x 23k rows that
dominates both runtime and peak memory.

Everything here works on ``np.bincount`` sums instead, so the cost is a handful
of vectorised passes per fold and the retained result is a ``models x groups``
float32 matrix - 0.2 MB at the scale above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

if TYPE_CHECKING:
    from drevalpy.types.results import ExperimentResult, ModelResult, RunResult

#: Groupings the comparison plots support, in report order.
GROUPINGS: Final[tuple[str, ...]] = ("drug", "cell_line")

#: Grouping name -> the ``RunResult`` attribute holding that grouping's labels.
_ID_ATTRIBUTE: Final[dict[str, str]] = {"drug": "drug_ids", "cell_line": "cell_line_ids"}

#: Human-readable axis/section labels per grouping.
GROUPING_LABELS: Final[dict[str, str]] = {"drug": "drug", "cell_line": "cell line"}

#: Correlation is undefined below this many observations in a group.
MIN_GROUP_SIZE: Final[int] = 2

Grouping = Literal["drug", "cell_line"]


def group_labels(run: RunResult, grouping: str) -> np.ndarray:
    """Return the grouping labels of a run.

    :param run: Run whose identifier arrays to read.
    :param grouping: One of :data:`GROUPINGS`.
    :returns: The run's ``drug_ids`` or ``cell_line_ids`` array.
    :raises ValueError: If ``grouping`` is not one of :data:`GROUPINGS`.
    """
    try:
        attribute = _ID_ATTRIBUTE[grouping]
    except KeyError:
        raise ValueError(f"Unknown grouping {grouping!r}; expected one of {GROUPINGS}") from None
    return np.asarray(getattr(run, attribute))


def _scoring_runs(model: ModelResult) -> list[RunResult]:
    """Return the non-randomized runs of a model."""
    return [run for run in model.runs if run.randomization is None]


class _PearsonSums:
    """Streaming sufficient statistics for a grouped Pearson correlation.

    Holds the six ``bincount`` accumulators (count, sum x, sum y, sum x^2,
    sum y^2, sum xy) for a fixed group axis, so folds can be folded in one at a
    time and never coexist in memory.
    """

    __slots__ = ("_count", "_sx", "_sxx", "_sxy", "_sy", "_syy", "n_groups")

    def __init__(self, n_groups: int) -> None:
        self.n_groups = n_groups
        self._count = np.zeros(n_groups, dtype=np.int64)
        self._sx = np.zeros(n_groups, dtype=np.float64)
        self._sy = np.zeros(n_groups, dtype=np.float64)
        self._sxx = np.zeros(n_groups, dtype=np.float64)
        self._syy = np.zeros(n_groups, dtype=np.float64)
        self._sxy = np.zeros(n_groups, dtype=np.float64)

    def add(self, codes: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
        """Accumulate one batch of observations.

        Rows with a negative code (no matching group) or a NaN on either axis
        are dropped.

        :param codes: Group index per observation.
        :param x: First variable, aligned with ``codes``.
        :param y: Second variable, aligned with ``codes``.
        """
        codes = np.asarray(codes)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        keep = (codes >= 0) & np.isfinite(x) & np.isfinite(y)
        if not keep.all():
            codes, x, y = codes[keep], x[keep], y[keep]
        if codes.size == 0:
            return
        n = self.n_groups
        self._count += np.bincount(codes, minlength=n)
        self._sx += np.bincount(codes, weights=x, minlength=n)
        self._sy += np.bincount(codes, weights=y, minlength=n)
        self._sxx += np.bincount(codes, weights=x * x, minlength=n)
        self._syy += np.bincount(codes, weights=y * y, minlength=n)
        self._sxy += np.bincount(codes, weights=x * y, minlength=n)

    def correlations(self, *, min_count: int = MIN_GROUP_SIZE) -> np.ndarray:
        """Reduce the accumulated sums to one correlation per group.

        Groups with fewer than ``min_count`` observations, or with no variance
        on either axis, yield NaN - mirroring
        :func:`drevalpy.evaluation.pearson`, which returns NaN for a constant
        target.

        :param min_count: Minimum observations required for a finite value.
        :returns: ``float64`` array of length ``n_groups``.
        """
        counts = self._count.astype(np.float64)
        enough = self._count >= max(min_count, MIN_GROUP_SIZE)
        safe = np.where(enough, counts, 1.0)

        cov = self._sxy - self._sx * self._sy / safe
        var_x = self._sxx - self._sx * self._sx / safe
        var_y = self._syy - self._sy * self._sy / safe

        # Rounding can drive a genuinely zero variance slightly negative.
        denominator = np.sqrt(np.clip(var_x, 0.0, None) * np.clip(var_y, 0.0, None))
        valid = enough & (denominator > 0.0)

        out = np.full(self.n_groups, np.nan, dtype=np.float64)
        np.divide(cov, denominator, out=out, where=valid)
        return np.clip(out, -1.0, 1.0, out=out)


def grouped_pearson(
    codes: np.ndarray,
    n_groups: int,
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_count: int = MIN_GROUP_SIZE,
) -> np.ndarray:
    """Compute the Pearson correlation of ``x`` against ``y`` within each group.

    Vectorised over groups: no Python-level loop and no intermediate object per
    observation.

    :param codes: Group index per observation; negative entries are dropped.
    :param n_groups: Size of the group axis, so empty groups keep their slot.
    :param x: First variable, aligned with ``codes``.
    :param y: Second variable, aligned with ``codes``.
    :param min_count: Minimum observations required for a finite value.
    :returns: ``float64`` array of length ``n_groups``, NaN where undefined.
    """
    sums = _PearsonSums(n_groups)
    sums.add(codes, x, y)
    return sums.correlations(min_count=min_count)


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
        """Number of drugs or cell lines on the column axis."""
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
        per-cell-line naive baselines do exactly that - has no defined
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
        sums = _PearsonSums(n_groups)
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
