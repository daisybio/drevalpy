"""Mutation cell-line featurizer."""

from __future__ import annotations


from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers.cell_line.omics.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "mutations",
    description="Binary mutation features for cell lines.",
    category="native",
    contract=FeatureKind.DENSE,
)
class MutationsCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Featurize raw mutation features."""

    _default_view = "mutations"
