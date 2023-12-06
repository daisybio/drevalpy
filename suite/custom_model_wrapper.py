from suite.model_wrapper import DRPModel, SingleDRPModel
from suite.data_wrapper import DrugResponseDataset, CellLineFeatureDataset, DrugFeatureDataset


class CustomModelWrapper(DRPModel):
    """
    Wrapper class to be implemented for a new custom model.
    """
    def __init__(self, model_name: str, target: str, *args, **kwargs):
        """
        Initializes the model wrapper.
        :param model_name: name of the model
        :param target: target value
        """
        super(CustomModelWrapper, self).__init__(model_name, target, *args, **kwargs)

    def train(self, cell_line_input: CellLineFeatureDataset, drug_input: DrugFeatureDataset, output: DrugResponseDataset):
        """
        Trains the model.
        :param cell_line_input: training data associated with the cell line input
        :param drug_input: training data associated with the drug input
        :param output: training data associated with the reponse output
        """
        raise NotImplementedError("train method not implemented")

    def evaluate(self, cell_line_input: CellLineFeatureDataset, drug_input: DrugFeatureDataset, output: DrugResponseDataset):
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

    def get_model_name(self):
        """
        Returns the model name.
        :return: model name
        """
        return self.model_name

    def get_target(self):
        """
        Returns the target value.
        :return: target value
        """
        return self.target

    @staticmethod
    def get_cell_line_features():
        """
        Transforms the cell line input data into a feature tensor that can be supplied to the train method.
        :return: CellLineFeatureDataset
        """
        raise NotImplementedError("get_cell_line_features method not implemented")

    @staticmethod
    def get_drug_features():
        """
        Transforms the drug input data into a feature tensor that can be supplied to the train method.
        :return: DrugFeatureDataset
        """
        raise NotImplementedError("get_drug_features method not implemented")