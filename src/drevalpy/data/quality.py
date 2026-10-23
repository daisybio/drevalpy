"""Curve-quality filtering for the response matrix.

The published datasets are refit with `CurveCurator
<https://github.com/kusterlab/curve_curator>`_ and ship **every** fitted curve,
including the ones the fit itself says are meaningless. The refit generation
stores ``pEC50`` as the response matrix, and a pEC50 exists for every curve that
converged, so ``~np.isnan(response_matrix)`` alone counts junk curves as usable
observations. Filtering on the quality metrics that ship alongside is what
separates a measured pair from a trustworthy one.

:class:`CurveQuality` is the entry point, and :func:`curve_quality_mask` a thin
wrapper over it. Every quality metric a dataset carries is a field, and ``None``
disables that check, so :meth:`CurveQuality.default` applies exactly the
CurveCurator-derived rule:

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

There is no capability check and no silent fallback: a rule that needs a layer
the dataset lacks raises :class:`KeyError`. :meth:`CurveQuality.off` is the
explicit opt-out; it needs no layers.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    from drevalpy.types.data.mudatalike import MuDataLike

#: Sentinel layer name for the response matrix itself. ``pEC50`` is stored as
#: ``X`` (see ``response.uns["x_column"]``), not as a layer, so the two pEC50
#: options resolve through ``response_matrix``.
_RESPONSE_MATRIX: Final = "pEC50"

_Comparison = Callable[[np.ndarray, float], np.ndarray]
_Transform = Callable[[np.ndarray], np.ndarray] | None

#: Threshold field of :class:`CurveQuality` -> (response layer,
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

#: Names of the numeric thresholds, in table order.
_THRESHOLD_NAMES: Final = tuple(_RULES)

#: Layer holding CurveCurator's own up/down/not verdict.
_REGULATION_LAYER: Final = "regulation"


@dataclass(frozen=True)
class CurveQuality:
    """A curve-quality rule: every threshold a pair's fitted curve must meet.

    The defaults are the ``alpha = 0.05`` and ``fc_lim = 0.45`` that
    :mod:`drevalpy.curation._fit` passes to CurveCurator, expressed as
    ``relevance_score >= -log10(alpha)`` and ``abs(fold_change) >= fc_lim``.
    Every other threshold is ``None``, meaning "do not check this metric".
    Use :meth:`default` for that rule and :meth:`off` for no rule at all.

    A metric that is NaN fails: a curve CurveCurator could not score is not a
    curve worth training on.

    :ivar min_relevance_score: Minimum SAM-corrected relevance score. This is the
        multiple-testing-corrected statistic; prefer it over *max_p_value*.
    :ivar min_abs_fold_change: Minimum absolute log2 curve fold change, i.e. the
        effect size. The layer is already log2, so no transform is applied
        beyond the absolute value.
    :ivar max_p_value: Maximum raw, **uncorrected** F-test p-value.
    :ivar min_log_p_value: Minimum ``-log10(p_value)``, the uncorrected p-value on
        a log scale.
    :ivar min_f_value: Minimum F statistic of the fit.
    :ivar min_f_value_sam: Minimum s0-corrected F statistic.
    :ivar min_r2: Minimum coefficient of determination of the fit.
    :ivar max_rmse: Maximum root-mean-square error of the fit.
    :ivar min_signal_quality: Minimum signal quality.
    :ivar min_abs_slope: Minimum absolute Hill slope.
    :ivar max_abs_slope: Maximum absolute Hill slope. Useful because a slope
        pinned at the fitting bound describes a step, which is usually an
        artefact rather than a dose response.
    :ivar min_front: Minimum fitted upper plateau.
    :ivar max_back: Maximum fitted lower plateau.
    :ivar min_pec50: Minimum pEC50, read from the response matrix rather than a
        layer. Together with *max_pec50* this brackets fits whose inflection
        point falls outside the tested dose range.
    :ivar max_pec50: Maximum pEC50.
    :ivar regulation: Keep only curves CurveCurator labelled with one of these,
        out of ``"up"``, ``"down"`` and ``"not"``. Stored sorted and
        de-duplicated.
    """

    # The CurveCurator-derived rule, on by default.
    min_relevance_score: float | None = 1.3010299956639813  # -log10(0.05), i.e. alpha
    min_abs_fold_change: float | None = 0.45  # fc_lim, already log2 in the layer
    # Significance, off by default.
    max_p_value: float | None = None
    min_log_p_value: float | None = None
    min_f_value: float | None = None
    min_f_value_sam: float | None = None
    # Goodness of fit, off by default.
    min_r2: float | None = None
    max_rmse: float | None = None
    min_signal_quality: float | None = None
    # Curve shape, off by default.
    min_abs_slope: float | None = None
    max_abs_slope: float | None = None
    min_front: float | None = None
    max_back: float | None = None
    min_pec50: float | None = None
    max_pec50: float | None = None
    # CurveCurator's own verdict, off by default.
    regulation: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Store *regulation* as a sorted tuple so equal rules compare and serialize equal."""
        if self.regulation is not None:
            labels = {self.regulation} if isinstance(self.regulation, str) else set(self.regulation)
            object.__setattr__(self, "regulation", tuple(sorted(labels)))

    @classmethod
    def default(cls) -> CurveQuality:
        """Return the CurveCurator-derived rule: relevance score and fold change only."""
        return cls()

    @classmethod
    def off(cls) -> CurveQuality:
        """No quality rule: every pair with a finite response is kept.

        :meth:`mask` then reads no response layers, so datasets without the
        quality metrics can be used.
        """
        return cls(**{name: None for name in _THRESHOLD_NAMES})

    def _thresholds(self) -> dict[str, float]:
        """Return the thresholds that are set, by name."""
        return {name: getattr(self, name) for name in _THRESHOLD_NAMES if getattr(self, name) is not None}

    def mask(self, dataset: MuDataLike) -> np.ndarray:
        """Mask of the pairs whose fitted curve meets every requested threshold.

        When no threshold and no *regulation* is set (see :meth:`off`), this is
        ``np.isfinite(dataset.response_matrix)`` and reads no response layers.

        :param dataset: Dataset (or any :class:`~drevalpy.types.data.mudatalike.MuDataLike`)
            whose response layers hold the quality metrics.
        :returns: Boolean array of shape ``(n_samples, n_drugs)``, True where the curve
            passes. Use ``~mask`` to blank the failing pairs of a response matrix.
        :raises KeyError: If a layer a requested threshold needs is not in the dataset.
        :raises ValueError: If *regulation* contains a label CurveCurator does not use.
        """
        thresholds = self._thresholds()
        if not thresholds and self.regulation is None:
            return np.isfinite(np.asarray(dataset.response_matrix))
        return _combine(dataset, thresholds, self.regulation)

    def to_metadata(self) -> dict[str, Any]:
        """Return the set thresholds as a JSON-safe dict; unset (``None``) ones are omitted.

        :returns: Mapping of threshold name to float, plus ``regulation`` as a list of
            labels when set. ``off()`` gives an empty dict.
        """
        metadata: dict[str, Any] = {name: float(value) for name, value in self._thresholds().items()}
        if self.regulation is not None:
            metadata["regulation"] = list(self.regulation)
        return metadata

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> CurveQuality:
        """Rebuild the rule :meth:`to_metadata` described.

        Names absent from *metadata* are unset, so ``{}`` is :meth:`off`, not
        :meth:`default`.

        :param metadata: Mapping as produced by :meth:`to_metadata`.
        :returns: The rule, equal to the one that produced *metadata*.
        :raises ValueError: If *metadata* holds a name that is not a threshold.
        """
        unknown = sorted(set(metadata) - {*_THRESHOLD_NAMES, "regulation"})
        if unknown:
            raise ValueError(f"Unknown curve-quality option(s) {unknown}. Valid: {sorted(_RULES)} and 'regulation'")
        return replace(cls.off(), **metadata)


def curve_quality_mask(dataset: MuDataLike, **options: Any) -> np.ndarray:
    """Mask of the pairs whose fitted curve meets every requested threshold.

    A thin wrapper over :meth:`CurveQuality.mask`. *options* are the fields of
    :class:`CurveQuality` and start from its defaults, so the default call
    applies exactly the CurveCurator-derived rule; passing ``None`` for a
    threshold disables that check.

    :param dataset: Dataset (or any :class:`~drevalpy.types.data.mudatalike.MuDataLike`)
        whose response layers hold the quality metrics.
    :param **options: Any :class:`CurveQuality` field, such as ``min_r2=0.8`` or
        ``min_relevance_score=None``.
    :returns: Boolean array of shape ``(n_samples, n_drugs)``, True where the curve passes.
    :raises KeyError: If a layer a requested threshold needs is not in the dataset.
    :raises ValueError: If *regulation* contains a label CurveCurator does not use.
    :raises TypeError: If an option is not a :class:`CurveQuality` field.
    """
    return CurveQuality(**options).mask(dataset)


def _combine(
    dataset: MuDataLike,
    thresholds: Mapping[str, float],
    regulation: Collection[str] | None,
) -> np.ndarray:
    """AND together every requested check, starting from "everything passes"."""
    mask = np.ones(dataset.response_matrix.shape, dtype=bool)
    for name, threshold in thresholds.items():
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
