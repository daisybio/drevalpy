"""SMILESVec drug featurizer for Precily."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "smilesvec",
    description="Precomputed SMILESVec drug embeddings for Precily.",
    category="general_purpose",
)
class SmilesVecDrugFeaturizer(ViewDrugFeaturizer):
    output_contract: ClassVar[FeatureContract] = FeatureContract(
        kind=FeatureKind.DENSE,
        view="smilesvec",
    )

    def __init__(self, *, view: str = "smilesvec") -> None:
        super().__init__(view=view)
