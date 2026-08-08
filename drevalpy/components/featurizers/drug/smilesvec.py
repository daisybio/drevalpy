"""SMILESVec drug featurizer for Precily."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.feature_tables import load_generic_csv


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

    @classmethod
    def load_features(cls, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load generated SMILESVec embeddings under the predictor block name.

        :param dataset_name: dataset name.
        :param kwargs: Keyword arguments.
        :returns: Result.
        """
        _ = cls, kwargs
        features = load_generic_csv(dataset_name, "drug_smilesvec", index_col="pubchem_id")
        for views in features.features.values():
            views["smilesvec"] = views.pop("drug_smilesvec")
        features.meta_info["smilesvec"] = features.meta_info.pop("drug_smilesvec")
        return features
