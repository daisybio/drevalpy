"""BIONIC featurizer for DIPK."""

from __future__ import annotations

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.predictors.literature.dipk.data_utils import load_bionic_features
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.datasets.dataset import FeatureDataset


@register_cell_line_featurizer(
    "bionic",
    description="Precomputed BIONIC cell-line features for DIPK.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class BionicCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Bionic cell line featurizer component."""

    _default_view = "bionic_features"

    @classmethod
    def load_features(cls, data_path: str, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load precomputed DIPK BIONIC features.

        :param data_path: data path.
        :param dataset_name: dataset name.
        :param kwargs: Keyword arguments.
        :returns: Result.
        :raises ValueError: Raised on invalid input.
        """
        _ = cls
        gene_add_num = kwargs.get("gene_add_num", 512)
        if not isinstance(gene_add_num, int):
            raise ValueError("gene_add_num must be an integer")
        return load_bionic_features(data_path, dataset_name, gene_add_num=gene_add_num)
