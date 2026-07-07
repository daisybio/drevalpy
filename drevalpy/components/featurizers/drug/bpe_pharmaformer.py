"""BPE PharmaFormer drug featurizer."""

from __future__ import annotations

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "bpePharmaformer",
    description="Precomputed BPE PharmaFormer token rows from the bpe_smiles view.",
    category="literature",
    template_repo_url="https://github.com/zhouyuru1205/PharmaFormer",
    citation_doi="10.1038/s41698-025-01082-6",
    deviations=(
        "Consumes precomputed BPE token rows from the bpe_smiles view; "
        "offline embedding generation is implemented in "
        "drevalpy.datasets.featurizer.create_pharmaformer_drug_embeddings."
    ),
    contract=FeatureKind.DENSE,
)
class BpePharmaformerDrugFeaturizer(ViewDrugFeaturizer):
    """Bpe pharmaformer drug featurizer component."""

    def __init__(self, *, view: str = "bpe_smiles") -> None:
        super().__init__(view=view)
