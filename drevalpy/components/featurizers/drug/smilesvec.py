"""SMILESVec drug featurizer for Precily."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "smilesvec",
    description="Precomputed SMILESVec drug embeddings for Precily.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class SmilesVecDrugFeaturizer(ViewDrugFeaturizer):
    """Smiles vec drug featurizer component."""

    input_views: ClassVar[tuple[str, ...]] = ("smilesvec",)

    def __init__(self, *, view: str = "smilesvec") -> None:
        """Initialize instance state.

        :param view: view.
        """
        super().__init__(view=view)
