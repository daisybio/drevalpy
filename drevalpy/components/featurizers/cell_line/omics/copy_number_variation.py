"""GISTIC copy-number variation cell-line featurizer."""

from __future__ import annotations


from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers.cell_line.omics.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "copyNumberVariationGistic",
    description="GISTIC copy-number variation features for cell lines.",
    category="native",
    contract=FeatureKind.DENSE,
)
class CopyNumberVariationGisticCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Featurize GISTIC copy-number variation."""

    _default_view = "copy_number_variation_gistic"
