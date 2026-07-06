"""SMILESVec drug featurizer for Precily."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "smilesvec",
    description="Precomputed SMILESVec drug embeddings for Precily.",
    category="general_purpose",
    contract=FeatureKind.DENSE,
)
class SmilesVecDrugFeaturizer(ViewDrugFeaturizer):
    """Smiles vec drug featurizer component."""

    def __init__(self, *, view: str = "smilesvec") -> None:
        super().__init__(view=view)
