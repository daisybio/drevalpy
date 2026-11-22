"""Curve-quality filtering for the response matrix.

Published screens ship every CurveCurator fit, including curves whose fit is meaningless.
:class:`CurveQuality` defines thresholds on the companion quality layers (defaulting to
``relevance_score >= -log10(0.05)`` and ``abs(fold_change) >= 0.45``, matching CurveCurator's
``regulation != 0``). A ``None`` threshold disables that check; :meth:`CurveQuality.off`
disables all checks and needs no quality layers.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    from drevalpy.types.data.dataset import Dataset

_Comparison = Callable[[np.ndarray, float], np.ndarray]
_Transform = Callable[[np.ndarray], np.ndarray] | None

#: Threshold field of :class:`CurveQuality` -> (layer, passing comparison, optional transform).
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
    "min_pec50": ("pEC50", operator.ge, None),
    "max_pec50": ("pEC50", operator.le, None),
}

_THRESHOLD_NAMES: Final = tuple(_RULES)
_REGULATION_LAYER: Final = "regulation"


@dataclass(frozen=True)
class CurveQuality:
    """Thresholds a pair's fitted curve must meet to be kept.

    Defaults match CurveCurator's ``alpha = 0.05`` (``relevance_score >= -log10(0.05)``) and
    ``fc_lim = 0.45`` (``abs(fold_change) >= 0.45``). Unset (``None``) thresholds are ignored,
    and NaN metric values always fail. Use :meth:`off` to keep every finite response without
    reading quality layers.
    """

    min_relevance_score: float | None = 1.3010299956639813  # -log10(0.05)
    min_abs_fold_change: float | None = 0.45
    max_p_value: float | None = None
    min_log_p_value: float | None = None
    min_f_value: float | None = None
    min_f_value_sam: float | None = None
    min_r2: float | None = None
    max_rmse: float | None = None
    min_signal_quality: float | None = None
    min_abs_slope: float | None = None
    max_abs_slope: float | None = None
    min_front: float | None = None
    max_back: float | None = None
    min_pec50: float | None = None
    max_pec50: float | None = None
    regulation: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Store *regulation* as a sorted, deduplicated tuple."""
        if self.regulation is not None:
            labels = {self.regulation} if isinstance(self.regulation, str) else set(self.regulation)
            object.__setattr__(self, "regulation", tuple(sorted(labels)))

    @classmethod
    def default(cls) -> CurveQuality:
        """Return the default CurveCurator-derived rule."""
        return cls()

    @classmethod
    def off(cls) -> CurveQuality:
        """Disable all quality checks; keep every finite response in the dataset's measure."""
        return cls(**{name: None for name in _THRESHOLD_NAMES})

    def _thresholds(self) -> dict[str, float]:
        """Return the active numeric thresholds by name."""
        return {name: getattr(self, name) for name in _THRESHOLD_NAMES if getattr(self, name) is not None}

    def mask(self, dataset: Dataset) -> np.ndarray:
        """Return a boolean ``(n_samples, n_drugs)`` mask of pairs passing every active check.

        It does not look at the response the models train on: whether that is finite is
        :func:`~drevalpy.data.splitters.observed_mask`'s check. With no active check (:meth:`off`)
        every pair passes and no layer is read. Missing layers raise ``KeyError`` (via
        ``dataset.get_response_layer``) and unknown regulation labels raise ``ValueError``.

        :param dataset: Dataset whose response layers hold the curve-quality metrics.
        :returns: Boolean mask, True where the curve passes.
        :raises ValueError: If ``regulation`` contains an unknown label.
        """
        passed = np.ones(dataset.response_matrix.shape, dtype=bool)
        for name, threshold in self._thresholds().items():
            layer, comparison, transform = _RULES[name]
            values = np.asarray(dataset.get_response_layer(layer), dtype=np.float64)
            if transform is not None:
                values = transform(values)
            passed &= comparison(values, threshold) & ~np.isnan(values)
        if self.regulation is not None:
            from drevalpy.curation._anndata import REGULATION_ENCODING

            unknown = sorted(set(self.regulation) - set(REGULATION_ENCODING))
            if unknown:
                raise ValueError(f"Unknown regulation label(s) {unknown}. Valid: {sorted(REGULATION_ENCODING)}")
            codes = np.asarray(dataset.get_response_layer(_REGULATION_LAYER), dtype=np.float64)
            wanted = [REGULATION_ENCODING[label] for label in self.regulation]
            passed &= np.isin(codes, wanted)
        return passed

    def to_metadata(self) -> dict[str, Any]:
        """Return the active checks as a JSON-safe dict (empty for :meth:`off`).

        :returns: Active threshold values, plus ``regulation`` when set.
        """
        metadata: dict[str, Any] = {name: float(value) for name, value in self._thresholds().items()}
        if self.regulation is not None:
            metadata["regulation"] = list(self.regulation)
        return metadata
