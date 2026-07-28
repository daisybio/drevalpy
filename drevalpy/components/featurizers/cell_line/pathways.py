"""Pathway featurizer for Precily."""

from __future__ import annotations

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "pathways",
    description="Precomputed GSVA pathway features for Precily.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class PathwaysCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Pathways cell line featurizer component."""

    _default_view = "pathways"
