"""Pathway featurizer for Precily."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.feature_tables import load_generic_csv


@register_cell_line_featurizer(
    "pathways",
    description="Precomputed GSVA pathway features for Precily.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class PathwaysCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Pathways cell line featurizer component."""

    input_views: ClassVar[tuple[str, ...]] = ("pathways",)

    @classmethod
    def load_features(cls, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load generated GSVA pathway features under the predictor block name.

        :param dataset_name: dataset name.
        :param kwargs: Keyword arguments.
        :returns: Result.
        """
        _ = cls, kwargs
        features = load_generic_csv(dataset_name, "pathway_features")
        for views in features.features.values():
            views["pathways"] = views.pop("pathway_features")
        features.meta_info["pathways"] = features.meta_info.pop("pathway_features")
        return features
