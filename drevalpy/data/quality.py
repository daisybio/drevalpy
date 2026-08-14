"""Curve-quality filtering for the response matrix.

The published datasets are refit with `CurveCurator
<https://github.com/kusterlab/curve_curator>`_ and ship **every** fitted curve,
including the ones the fit itself says are meaningless. The refit generation
stores ``pEC50`` as the response matrix, and a pEC50 exists for every curve that
converged, so ``~np.isnan(response_matrix)`` alone counts junk curves as usable
observations. Filtering on the quality metrics that ship alongside is what
separates a measured pair from a trustworthy one.

:func:`curve_quality_mask` is the single entry point. Every quality metric a
dataset carries is a keyword option, and passing ``None`` disables that check,
so the default call applies exactly the CurveCurator-derived rule:

.. code-block:: python

    relevance_score >= -log10(0.05) and abs(fold_change) >= 0.45

Those two numbers are the ``alpha`` and ``fc_lim`` that
:mod:`drevalpy.curation._fit` passes to CurveCurator's ``F Statistic`` block.
Upstream they only *annotate* - ``apply_significance_thresholds`` adds columns
and drops no rows - but they are not inert: they determine the ``s0`` fudge
factor behind ``relevance_score``, and they define the ``regulation`` label. The
default rule therefore reproduces ``regulation != 0`` exactly.

Two choices worth stating, because the obvious alternatives are wrong:

* **Gate on ``relevance_score``, not ``p_value``.** ``p_value`` is the raw,
  uncorrected F-test p-value. The multiple-testing control lives in
  ``relevance_score``, which is the SAM-corrected statistic. Thresholding
  ``p_value`` across the tens of thousands of curves in a screen would apply no
  correction at all. It is exposed, but off by default.
* **Recompute rather than read ``regulation``.** It gives identical results by
  default, but it is NaN wherever CurveCurator reached no verdict and it encodes
  direction rather than quality. It is exposed as a categorical option.

There is no capability check and no silent fallback: the layers are guaranteed
by the file format, so a missing one raises :class:`KeyError`.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from drevalpy.types.data.mudatalike import MuDataLike

#: Sentinel layer name for the response matrix itself. ``pEC50`` is stored as
#: ``X`` (see ``response.uns["x_column"]``), not as a layer, so the two pEC50
#: options resolve through ``response_matrix``.
_RESPONSE_MATRIX: Final = "pEC50"

_Comparison = Callable[[np.ndarray, float], np.ndarray]
_Transform = Callable[[np.ndarray], np.ndarray] | None

#: Keyword argument of :func:`curve_quality_mask` -> (response layer,
#: comparison that a *passing* curve satisfies, optional value transform).
#:
#: Driving the checks from a table rather than a branch per metric is what keeps
#: :func:`curve_quality_mask` flat, and it lets the tests parametrize over every
#: option so a new rule cannot land untested.
_RULES: Final[dict[str, tuple[str, _Comparison, _Transform]]] = {
    "min_relevance_score": ("relevance_score", operator.ge, None),
    "min_abs_fold_change": ("fold_change", operator.ge, np.abs),
    "max_p_value": ("p_value", operator.le, None),
    "min_log_p_value": ("log_p_value", operator.ge, None),
    "min_f_value": ("f_value", operator.ge, None),
    "min_f_value_sam": ("f_value_sam", operator.ge, None),
    "min_r2": ("R2", operator.ge, None),
    "max_rmse": ("RMSE", operator.le, None),
    "min_signal_quality": ("signal_quality", operator.ge, None),
    "min_abs_slope": ("slope", operator.ge, np.abs),
    "max_abs_slope": ("slope", operator.le, np.abs),
    "min_front": ("front", operator.ge, None),
    "max_back": ("back", operator.le, None),
    "min_pec50": (_RESPONSE_MATRIX, operator.ge, None),
    "max_pec50": (_RESPONSE_MATRIX, operator.le, None),
}

#: Layer holding CurveCurator's own up/down/not verdict.
_REGULATION_LAYER: Final = "regulation"


def curve_quality_mask(
    dataset: MuDataLike,
    *,
    # The CurveCurator-derived rule, on by default.
    min_relevance_score: float | None = 1.3010299956639813,  # -log10(0.05), i.e. alpha
    min_abs_fold_change: float | None = 0.45,  # fc_lim, already log2 in the layer
    # Significance, off by default.
    max_p_value: float | None = None,
    min_log_p_value: float | None = None,
    min_f_value: float | None = None,
    min_f_value_sam: float | None = None,
    # Goodness of fit, off by default.
    min_r2: float | None = None,
    max_rmse: float | None = None,
    min_signal_quality: float | None = None,
    # Curve shape, off by default.
    min_abs_slope: float | None = None,
    max_abs_slope: float | None = None,
    min_front: float | None = None,
    max_back: float | None = None,
    min_pec50: float | None = None,
    max_pec50: float | None = None,
    # CurveCurator's own verdict, off by default.
    regulation: Collection[str] | None = None,
) -> np.ndarray:
    """Mask of the pairs whose fitted curve meets every requested threshold.

    The defaults are the ``alpha = 0.05`` and ``fc_lim = 0.45`` that
    :mod:`drevalpy.curation._fit` passes to CurveCurator, expressed as
    ``relevance_score >= -log10(alpha)`` and ``abs(fold_change) >= fc_lim``.
    Every other option is ``None``, meaning "do not check this metric".

    A metric that is NaN fails: a curve CurveCurator could not score is not a
    curve worth training on.

    Args:
        dataset: Dataset (or any :class:`~drevalpy.types.data.mudatalike.MuDataLike`)
            whose response layers hold the quality metrics.
        min_relevance_score: Minimum SAM-corrected relevance score. This is the
            multiple-testing-corrected statistic; prefer it over *max_p_value*.
        min_abs_fold_change: Minimum absolute log2 curve fold change, i.e. the
            effect size. The layer is already log2, so no transform is applied
            beyond the absolute value.
        max_p_value: Maximum raw, **uncorrected** F-test p-value.
        min_log_p_value: Minimum ``-log10(p_value)``, the uncorrected p-value on
            a log scale.
        min_f_value: Minimum F statistic of the fit.
        min_f_value_sam: Minimum s0-corrected F statistic.
        min_r2: Minimum coefficient of determination of the fit.
        max_rmse: Maximum root-mean-square error of the fit.
        min_signal_quality: Minimum signal quality.
        min_abs_slope: Minimum absolute Hill slope.
        max_abs_slope: Maximum absolute Hill slope. Useful because a slope
            pinned at the fitting bound describes a step, which is usually an
            artefact rather than a dose response.
        min_front: Minimum fitted upper plateau.
        max_back: Maximum fitted lower plateau.
        min_pec50: Minimum pEC50, read from the response matrix rather than a
            layer. Together with *max_pec50* this brackets fits whose inflection
            point falls outside the tested dose range.
        max_pec50: Maximum pEC50.
        regulation: Keep only curves CurveCurator labelled with one of these,
            out of ``"up"``, ``"down"`` and ``"not"``.

    Returns:
        Boolean array of shape ``(n_cell_lines, n_drugs)``, True where the curve
        passes. Use ``~mask`` to blank the failing pairs of a response matrix.

    Raises:
        KeyError: If a layer a requested threshold needs is not in the dataset.
        ValueError: If *regulation* contains a label CurveCurator does not use.
    """
    thresholds: dict[str, float | None] = {
        "min_relevance_score": min_relevance_score,
        "min_abs_fold_change": min_abs_fold_change,
        "max_p_value": max_p_value,
        "min_log_p_value": min_log_p_value,
        "min_f_value": min_f_value,
        "min_f_value_sam": min_f_value_sam,
        "min_r2": min_r2,
        "max_rmse": max_rmse,
        "min_signal_quality": min_signal_quality,
        "min_abs_slope": min_abs_slope,
        "max_abs_slope": max_abs_slope,
        "min_front": min_front,
        "max_back": max_back,
        "min_pec50": min_pec50,
        "max_pec50": max_pec50,
    }
    return _combine(dataset, thresholds, regulation)


def _combine(
    dataset: MuDataLike,
    thresholds: dict[str, float | None],
    regulation: Collection[str] | None,
) -> np.ndarray:
    """AND together every requested check, starting from "everything passes"."""
    mask = np.ones(dataset.response_matrix.shape, dtype=bool)
    for name, threshold in thresholds.items():
        if threshold is not None:
            mask &= _threshold_mask(dataset, name, threshold)
    if regulation is not None:
        mask &= _regulation_mask(dataset, regulation)
    return mask


def _threshold_mask(dataset: MuDataLike, name: str, threshold: float) -> np.ndarray:
    """Apply one row of :data:`_RULES`."""
    layer, comparison, transform = _RULES[name]
    values = _metric(dataset, layer)
    if transform is not None:
        values = transform(values)
    # A NaN comparison is already False for both directions, but say it outright:
    # "no score" must never read as "passed".
    return comparison(values, threshold) & ~np.isnan(values)


def _metric(dataset: MuDataLike, layer: str) -> np.ndarray:
    """Read one metric, resolving the pEC50 sentinel to the response matrix."""
    source = dataset.response_matrix if layer == _RESPONSE_MATRIX else dataset.get_response_layer(layer)
    return np.asarray(source, dtype=np.float64)


def _regulation_mask(dataset: MuDataLike, labels: Collection[str]) -> np.ndarray:
    """Keep pairs whose ``regulation`` layer matches one of *labels*.

    The layer is numeric because an AnnData layer has to be; the encoding lives
    with the code that writes it, so it is imported rather than restated. That
    import is deferred because :mod:`drevalpy.curation._anndata` pulls in
    ``anndata`` and ``pandas``, and this module is on the critical path of
    ``import drevalpy`` via the splitter registration.
    """
    from drevalpy.curation._anndata import _REGULATION_ENCODING

    unknown = sorted(set(labels) - set(_REGULATION_ENCODING))
    if unknown:
        raise ValueError(f"Unknown regulation label(s) {unknown}. Valid: {sorted(_REGULATION_ENCODING)}")

    wanted = [_REGULATION_ENCODING[label] for label in labels]
    values = np.asarray(dataset.get_response_layer(_REGULATION_LAYER), dtype=np.float64)
    # NaN is in no category, so an undetermined curve is excluded.
    return np.isin(values, wanted)
