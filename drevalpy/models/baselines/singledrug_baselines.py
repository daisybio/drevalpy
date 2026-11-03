"""SingleDrugElasticNet, SingleDrugRandomForest and SingleDrugKNN class. Fit a model for each drug separately."""

from sklearn.pipeline import Pipeline

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..utils import MRMRSelector
from .sklearn_models import ElasticNetModel, KNNRegressor, RandomForest


class SingleDrugElasticNet(ElasticNetModel):
    """SingleDrugElasticNet class."""

    is_single_drug_model = True
    drug_views = []
    early_stopping = False

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SingleDrugElasticNet
        """
        return "SingleDrugElasticNet"

    def build_model(self, hyperparameters: dict):
        """
        Overwrites drug views to be empty.

        :param hyperparameters: hyperparameters
        """
        super().build_model(hyperparameters)
        self.drug_views = []

    def load_drug_features(self, data_path, dataset_name):
        """
        Load drug features. Not needed for SingleDrugElasticNet.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: None
        """
        return None


class SingleDrugRandomForest(RandomForest):
    """SingleDrugRandomForest class."""

    is_single_drug_model = True
    drug_views = []
    early_stopping = False

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SingleDrugRandomForest
        """
        return "SingleDrugRandomForest"

    def build_model(self, hyperparameters: dict):
        """
        Overwrites drug views to be empty.

        :param hyperparameters: hyperparameters
        """
        super().build_model(hyperparameters)
        self.drug_views = []

    def load_drug_features(self, data_path, dataset_name):
        """
        Load drug features. Not needed for SingleDrugRandomForest.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: None
        """
        return None


class SingleDrugKNN(KNNRegressor):
    """
    SingleDrugKNN class.

    For a test cell line, the most similar training cell lines are searched and their responses to the same drug are
    averaged. The distance is computed on the ``mrmr_k`` genes that are most relevant for the response of this drug
    (mRMR selection, fitted on the training rows only), not on the full expression profile.
    """

    is_single_drug_model = True
    drug_views = []
    early_stopping = False

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SingleDrugKNN
        """
        return "SingleDrugKNN"

    def build_model(self, hyperparameters: dict):
        """
        Overwrites drug views to be empty and puts the mRMR selection in front of the kNN regressor.

        :param hyperparameters: hyperparameters, ``mrmr_k`` is the number of selected features (None or 0 = no
            selection)
        """
        super().build_model(hyperparameters)
        self.drug_views = []
        mrmr_k = hyperparameters.get("mrmr_k", 20)
        selector = MRMRSelector(k=int(mrmr_k)) if mrmr_k else "passthrough"
        self.model = Pipeline([("mrmr", selector), ("knn", self.model)])

    def load_drug_features(self, data_path, dataset_name):
        """
        Load drug features. Not needed for SingleDrugKNN.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: None
        """
        return None

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Trains the model; with fewer training cell lines than neighbors, the number of neighbors is reduced.

        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing gene expression data
        :param drug_input: not needed
        :param output_earlystopping: not needed
        :param model_checkpoint_dir: not needed
        """
        if 0 < len(output) < self.hyperparameters["n_neighbors"]:
            self.model.set_params(knn__n_neighbors=len(output))
        super().train(output, cell_line_input, drug_input, output_earlystopping, model_checkpoint_dir)
