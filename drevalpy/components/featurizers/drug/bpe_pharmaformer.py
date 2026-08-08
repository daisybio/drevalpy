"""BPE PharmaFormer drug featurizer."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import BlockSpec
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.types.literature_reference import LiteratureReference

_BPE_PHARMAFORMER_REFERENCE = LiteratureReference(
    repo_url="https://github.com/zhouyuru1205/PharmaFormer",
    citation_doi="10.1038/s41698-025-01082-6",
    deviations=(
        "Consumes precomputed BPE token rows from the bpe_smiles view; "
        "offline embedding generation is implemented in "
        "drevalpy.datasets.featurizer.create_pharmaformer_drug_embeddings."
    ),
)


@register_drug_featurizer(
    "bpePharmaformer",
    description="Precomputed BPE PharmaFormer token rows from the bpe_smiles view.",
    reference=_BPE_PHARMAFORMER_REFERENCE,
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class BpePharmaformerDrugFeaturizer(ViewDrugFeaturizer):
    """BPE PharmaFormer drug featurizer component."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("bpe_smiles", FeatureFormat.NUMERIC_MATRIX),)
    input_views: ClassVar[tuple[str, ...]] = ("bpe_smiles",)

    def __init__(self, *, view: str = "bpe_smiles") -> None:
        """Initialize instance state.

        :param view: view.
        """
        super().__init__(view=view)
