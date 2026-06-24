"""Mutation cell-line featurizer."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.featurizers.cell_line.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "mutations",
    description="Binary mutation features for cell lines.",
    category="native",
)
class MutationsCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Featurize raw mutation features."""

    _default_view = "mutations"
    output_contract: ClassVar[FeatureContract] = FeatureContract(
        kind=FeatureKind.DENSE,
        view="mutations",
    )
