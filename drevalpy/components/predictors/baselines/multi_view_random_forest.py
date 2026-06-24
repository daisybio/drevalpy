"""Contains the Multi-OMICS Random Forest model."""

from drevalpy.models.utils import load_multi_cell_line_view

from .sklearn_models import RandomForest


class MultiViewRandomForest(RandomForest):
    """Multi-View Random Forest model."""

    cell_line_views = [
        "gene_expression",
        "methylation",
        "mutations",
        "copy_number_variation_gistic",
    ]

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: MultiViewRandomForest
        """
        return "MultiViewRandomForest"

    def load_cell_line_features(self, data_path: str, dataset_name: str):
        """
        Loads the cell line features for a multi-view random forest.

        :param data_path: data path e.g. data/
        :param dataset_name: dataset name e.g. GDSC1
        :returns: FeatureDataset containing the cell line omics features
        """
        return load_multi_cell_line_view(self.cell_line_views, data_path, dataset_name, self.get_model_name())
