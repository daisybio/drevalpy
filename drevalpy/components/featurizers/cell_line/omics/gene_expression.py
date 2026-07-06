"""Raw gene-expression cell-line featurizer."""

from __future__ import annotations


from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers.cell_line.omics.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "geneExpression",
    description="Raw landmark gene expression without additional preprocessing.",
    category="native",
    contract=FeatureKind.DENSE,
)
class GeneExpressionCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Featurize raw gene expression."""

    _default_view = "gene_expression"
