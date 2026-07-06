"""Pathway featurizer for Precily."""

from __future__ import annotations

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers.cell_line.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "pathways",
    description="Precomputed GSVA pathway features for Precily.",
    category="general_purpose",
    contract=FeatureKind.DENSE,
)
class PathwaysCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Pathways cell line featurizer component."""

    _default_view = "pathways"
