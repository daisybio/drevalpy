"""Concatenate outputs from multiple cell-line featurizers."""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "concatFeaturizers",
    description="Concatenate dense outputs from multiple cell-line featurizers.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ConcatFeaturizersCellLineFeaturizer(ConcatFeaturizersMixin, CellLineFeaturizer):
    """Fit child featurizers independently and concatenate their dense outputs."""

    _not_fitted_msg = "ConcatFeaturizersCellLineFeaturizer must be fit before transform"

    def __init__(
        self,
        *,
        featurizers: list[Any] | None = None,
    ) -> None:
        """Initialize instance state.

        :param featurizers: featurizers.
        """
        self._init_concat(featurizers=featurizers, registry="cell_line")
