from suite.model_wrapper import DRPModel
from suite.data_wrapper import DrugResponseDataset, FeatureDataset


class CustomModelWrapper(DRPModel):
    """
    Wrapper class to be implemented for a new custom model.
    """
    def build_model(self, *args, **kwargs):
        pass

    def train(self, cell_line_input: FeatureDataset, drug_input: FeatureDataset, output: DrugResponseDataset):
        """
        Trains the model.
        :param cell_line_input: training data associated with the cell line input
        :param drug_input: training data associated with the drug input
        :param output: training data associated with the reponse output
        """
        raise NotImplementedError("train method not implemented")

    def predict(self, cell_line_input: FeatureDataset, drug_input: FeatureDataset):
        pass

    def evaluate(self, cell_line_input: FeatureDataset, drug_input: FeatureDataset, output: DrugResponseDataset):
        """
        Evaluates the model.
        :param cell_line_input: evaluation data associated with the cell line input
        :param drug_input: evaluation data associated with the drug input
        :param output: evaluation data associated with the reponse output
        """
        raise NotImplementedError("evaluate method not implemented")

    def save(self, path):
        """
        Saves the model.
        :param path: path to save the model
        """
        raise NotImplementedError("save method not implemented")

    def load(self, path):
        """
        Loads the model.
        :param path: path to load the model
        """
        raise NotImplementedError("load method not implemented")

    def get_cell_line_features(self, cell_line_input: FeatureDataset):
        """
        Transforms the cell line input data into a feature tensor that can be supplied to the train method.
        :return: CellLineFeatureDataset
        """
        raise NotImplementedError("get_cell_line_features method not implemented")

    def get_drug_features(self, drug_input: FeatureDataset):
        """
        Transforms the drug input data into a feature tensor that can be supplied to the train method.
        :return: DrugFeatureDataset
        """
        raise NotImplementedError("get_drug_features method not implemented")