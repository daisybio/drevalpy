"""SMILESVec drug featurizer for Precily."""

from __future__ import annotations

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.data.features import load_generic_csv
from drevalpy.datasets.dataset import FeatureDataset


@register_drug_featurizer(
    "smilesvec",
    description="Precomputed SMILESVec drug embeddings for Precily.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class SmilesVecDrugFeaturizer(ViewDrugFeaturizer):
    """Smiles vec drug featurizer component."""

    def __init__(self, *, view: str = "smilesvec") -> None:
        super().__init__(view=view)

    @classmethod
    def load_features(cls, data_path: str, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load generated SMILESVec embeddings under the predictor block name."""
        _ = cls, kwargs
        features = load_generic_csv(data_path, dataset_name, "drug_smilesvec", index_col="pubchem_id")
        for views in features.features.values():
            views["smilesvec"] = views.pop("drug_smilesvec")
        features.meta_info["smilesvec"] = features.meta_info.pop("drug_smilesvec")
        return features
