"""Cell-line constant (one-category / intercept) featurizer."""

from __future__ import annotations

from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._constant import ConstantFeaturizerMixin
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "constant",
    description="Constant one-column intercept features with no cell-line identity.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class CellLineConstantFeaturizer(ConstantFeaturizerMixin, CellLineFeaturizer):
    """Emit ones for every cell-line entity."""
