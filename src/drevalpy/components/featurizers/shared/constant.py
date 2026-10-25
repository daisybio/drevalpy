"""Constant (one-category / intercept) featurizer, shared by both entity sides."""

from __future__ import annotations

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._constant import ConstantFeaturizerMixin
from drevalpy.components.featurizers._side_binding import register_for_sides
from drevalpy.components.featurizers.base import Featurizer


@register_for_sides(
    "constant",
    description={
        "sample": "Constant one-column intercept features with no sample identity.",
        "drug": "Constant one-column intercept features with no drug identity.",
    },
    output_format=FeatureFormat.NUMERIC_MATRIX,
)
class SharedConstantFeaturizer(ConstantFeaturizerMixin, Featurizer):
    """Emit ones for every entity, on either side."""
