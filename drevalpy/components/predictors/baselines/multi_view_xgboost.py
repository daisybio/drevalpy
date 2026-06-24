"""Contains the baseline MultiViewXGBoost model."""

from drevalpy.datasets.dataset import FeatureDataset

from drevalpy.models.utils import load_multi_cell_line_view, load_single_drug_view

from .sklearn_models import SklearnModel


class MultiViewXGBoost(SklearnModel):
    """XGBoost model with multi-omic cell line features and drug fingerprints."""

    cell_line_views = [
        "gene_expression",
        "methylation",
        "mutations",
        "copy_number_variation_gistic",
    ]
    drug_views = ["fingerprints"]

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: MultiViewXGBoost
        """
        return "MultiViewXGBoost"

    def build_model(self, hyperparameters: dict) -> None:
        """
        Builds the model from hyperparameters.

        :param hyperparameters: dictionary containing the hyperparameters.
        :raises ImportError: if xgboost is not installed.
        """
        try:
            import xgboost  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "MultiViewXGBoost requires the optional 'xgboost' extra. "
                "Install it with: pip install drevalpy[xgboost] (or `poetry install -E xgboost`)."
            ) from e

        merged = dict(hyperparameters)
        merged.setdefault("cell_line_views", self.cell_line_views)
        merged.setdefault("drug_views", self.drug_views)
        super().build_model(merged)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: data path e.g. data/
        :param dataset_name: dataset name e.g. GDSC1
        :returns: FeatureDataset containing the cell line omics features
        """
        return load_multi_cell_line_view(self.cell_line_views, data_path, dataset_name, self.get_model_name())

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Loads the drug features.

        :param data_path: path to the drug features, e.g., data/
        :param dataset_name: name of the dataset, e.g., GDSC1
        :returns: FeatureDataset containing the drug features
        """
        return load_single_drug_view(self.drug_views, data_path, dataset_name, self.get_model_name())
