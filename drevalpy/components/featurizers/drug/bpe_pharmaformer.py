"""BPE PharmaFormer drug featurizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.datasets.dataset import FeatureDataset
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

    def __init__(self, *, view: str = "bpe_smiles") -> None:
        """Initialize instance state.

        :param view: view.
        """
        super().__init__(view=view)

    @classmethod
    def load_features(cls, data_path: str, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load precomputed PharmaFormer BPE token embeddings.

        :param data_path: data path.
        :param dataset_name: dataset name.
        :param kwargs: Keyword arguments.
        :returns: Result.
        :raises FileNotFoundError: Raised on invalid input.
        """
        _ = cls, kwargs
        path = Path(data_path) / dataset_name / "drug_bpe_smiles.csv"
        if not path.exists():
            raise FileNotFoundError(f"BPE SMILES file not found: {path}")
        frame = pd.read_csv(path, dtype={"pubchem_id": str})
        return FeatureDataset(
            {
                str(row["pubchem_id"]): {"bpe_smiles": row.drop("pubchem_id").to_numpy(dtype=np.float32)}
                for _, row in frame.iterrows()
            }
        )
