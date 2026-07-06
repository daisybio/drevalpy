"""BIONIC featurizer for DIPK."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.featurizers.cell_line.omics.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "bionic",
    description="Precomputed BIONIC cell-line features for DIPK.",
    category="general_purpose",
)
class BionicCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Bionic cell line featurizer component."""

    output_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE)

    _default_view = "bionic_features"
