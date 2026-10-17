"""Constant (one-category / intercept) featurizer, shared by both entity sides."""

from __future__ import annotations

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._constant import ConstantFeaturizerMixin
from drevalpy.components.featurizers._side_binding import register_for_sides
from drevalpy.components.featurizers.base import Featurizer


@register_for_sides(
    "constant",
    description={
        "cell_line": "Constant one-column intercept features with no cell-line identity.",
        "drug": "Constant one-column intercept features with no drug identity.",
    },
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class SharedConstantFeaturizer(ConstantFeaturizerMixin, Featurizer):
    """Emit ones for every entity, on either side."""
