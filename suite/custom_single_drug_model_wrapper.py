from suite.model_wrapper import SingleDRPModel
from suite.data_wrapper import DrugResponseDataset, FeatureDataset


class CustomSingleDrugModelWrapper(SingleDRPModel):
    """
    Wrapper class to be implemented for a new custom model that builds one model per drug.
    """

    @property
    def cell_line_views(self):
        """
        Returns the sources the model needs as input for describing the cell line.
        :return: cell line views, e.g., ["methylation", "gene_expression", "mirna_expression", "mutation"]
        """
        return []

    @property
    def drug_views(self):
        """
        Returns the sources the model needs as input for describing the drug.
        :return: drug views, e.g., ["descriptors", "fingerprints", "targets"]
        """
        return []

    def build_model(self, *args, **kwargs):
        """
        Builds the model.
        """
        pass

    def train_drug(
        self,
        cell_line_input: FeatureDataset,
        drug_name: str,
        output: DrugResponseDataset,
    ):
        """
        Trains one model per drug.
        :param cell_line_input: training data associated with the cell line input
        :param drug_name: drug name
        :param output: training data associated with the response output
        """
        pass

    def predict_drug(self, cell_line_input: FeatureDataset, drug_name: str):
        """
        Predicts the response for the given single drug.
        :param cell_line_input: input associated with the cell line
        :param drug_name: drug name
        :return: predicted response
        """
        pass

    def evaluate(
        self,
        cell_line_input: FeatureDataset,
        drug_input: str,
        output: DrugResponseDataset,
    ):
        """
        Evaluates the model.
        :param cell_line_input: evaluation data associated with the cell line input
        :param drug_input: evaluation data associated with the drug input
        :param output: evaluation data associated with the response output
        """
        pass

    def save(self, path):
        """
        Saves the model to models_saved.
        :param path: path to save the model
        """
        pass

    def load(self, path):
        """
        Loads the model.
        :param path: path to load the model
        """
        pass

    def transform_cell_line_features(self, cell_line_input: FeatureDataset):
        """
        Transforms the cell line input data into a feature tensor that can be supplied to the train method.
        :return: FeatureDataset
        """
        pass

    def transform_drug_features(self, drug_input: FeatureDataset):
        """
        Transforms the drug input data into a feature tensor that can be supplied to the train method.
        :return: FeatureDataset
        """
        pass
