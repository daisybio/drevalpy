"""Cell-line constant (one-category / intercept) featurizer."""

from __future__ import annotations

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._constant import ConstantFeaturizerMixin
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.registry.cell_line_featurizer import register


@register(
    "constant",
    description="Constant one-column intercept features with no cell-line identity.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class CellLineConstantFeaturizer(ConstantFeaturizerMixin, CellLineFeaturizer):
    """Emit ones for every cell-line entity."""
