from suite.model_wrapper import DRPModel
from suite.data_wrapper import DrugResponseDataset, FeatureDataset


class CustomModelWrapper(DRPModel):
    """
    Wrapper class to be implemented for a new custom model.
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

    def train(self, cell_line_input: FeatureDataset, drug_input: FeatureDataset, output: DrugResponseDataset):
        """
        Trains the model.
        :param cell_line_input: training data associated with the cell line input
        :param drug_input: training data associated with the drug input
        :param output: training data associated with the reponse output
        """
        pass

    def predict(self, cell_line_input: FeatureDataset, drug_input: FeatureDataset):
        """
        Predicts the response for the given input. Call the respective function from models_code here.
        :param cell_line_input: input associated with the cell line
        :param drug_input: input associated with the drug
        :return: predicted response
        """
        pass

    def evaluate(self, cell_line_input: FeatureDataset, drug_input: FeatureDataset, output: DrugResponseDataset):
        """
        Evaluates the model.
        :param cell_line_input: evaluation data associated with the cell line input
        :param drug_input: evaluation data associated with the drug input
        :param output: evaluation data associated with the response output
        """
        pass

    def save(self, path):
        """
        Saves the model.
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
