"""BPE PharmaFormer drug featurizer."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "bpePharmaformer",
    description="Precomputed BPE PharmaFormer token rows from the bpe_smiles view.",
    category="general_purpose",
)
class BpePharmaformerDrugFeaturizer(ViewDrugFeaturizer):
    """Bpe pharmaformer drug featurizer component."""

    output_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE)

    def __init__(self, *, view: str = "bpe_smiles") -> None:
        super().__init__(view=view)
