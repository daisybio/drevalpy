"""Raw gene-expression cell-line featurizer."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.featurizers.cell_line.omics.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "geneExpression",
    description="Raw landmark gene expression without additional preprocessing.",
    category="native",
)
class GeneExpressionCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Featurize raw gene expression."""

    _default_view = "gene_expression"
    output_contract: ClassVar[FeatureContract] = FeatureContract(
        kind=FeatureKind.DENSE,
        view="gene_expression",
    )
