"""BIONIC featurizer for DIPK."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.featurizers.cell_line.view import ViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "bionic",
    description="Precomputed BIONIC cell-line features for DIPK.",
    category="general_purpose",
)
class BionicCellLineFeaturizer(ViewCellLineFeaturizer):
    output_contract: ClassVar[FeatureContract] = FeatureContract(
        kind=FeatureKind.DENSE,
        view="bionic_features",
    )

    def __init__(self, *, view: str = "bionic_features") -> None:
        super().__init__(view=view)
