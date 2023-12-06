from abc import ABC, abstractmethod
from data_wrapper import DrugResponseDataset, FeatureDataset


class DRPModel(ABC):
    """
    Abstract wrapper class for drug response prediction models.
    """

    def __init__(self, model_name, target, *args, **kwargs):
        """
        Creates an instance of a drug response prediction model.
        :param model_name: model name for displaying results
        :param target: target value, e.g., IC50, EC50, AUC, classification
        :param args: optional arguments
        :param kwargs: optional keyword arguments
        """
        self.model_name = model_name
        self.target = target
        self.build_model(*args, **kwargs)

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """
        Builds the model.
        """
        pass

    @abstractmethod
    def train(self, cell_line_input: FeatureDataset, drug_input: FeatureDataset, output: DrugResponseDataset):
        """
        Trains the model. Call the respective function from models_code here.
        :param cell_line_input: training data associated with the cell line input
        :param drug_input: training data associated with the drug input
        :param output: training data associated with the response output
        """
        pass

    @abstractmethod
    def predict(self, cell_line_input: FeatureDataset, drug_input: FeatureDataset):
        """
        Predicts the response for the given input. Call the respective function from models_code here.
        :param cell_line_input: input associated with the cell line
        :param drug_input: input associated with the drug
        :return: predicted response
        """
        pass

    @abstractmethod
    def evaluate(self, cell_line_input: FeatureDataset, drug_input: FeatureDataset, output: DrugResponseDataset):
        """
        Evaluates the model. Call the respective function(s) from models_code here.
        :param cell_line_input: evaluation data associated with the cell line input
        :param drug_input: evaluation data associated with the drug input
        :param output: evaluation data associated with the reponse output
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Saves the model to models_saved.
        :param path: path to save the model
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Loads the model.
        :param path: path to load the model
        """
        pass

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
        :return: FeatureDataset
        """
        raise NotImplementedError("get_cell_line_features method not implemented")

    @staticmethod
    def get_drug_features():
        """
        Transforms the drug input data into a feature tensor that can be supplied to the train method.
        :return: FeatureDataset
        """
        raise NotImplementedError("get_drug_features method not implemented")


class SingleDRPModel(DRPModel, ABC):
    """
    Abstract wrapper class for single drug response prediction models.
    """

    def __init__(self, model_name, target, *args, **kwargs):
        """
        Creates an instance of a single drug response prediction model.
        :param model_name: model name for displaying results
        :param target: target value, e.g., IC50, EC50, AUC, classification
        :param args: optional arguments
        :param kwargs: optional keyword arguments
        """
        super(SingleDRPModel, self).__init__(model_name, target, *args, **kwargs)

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """
        Builds the model.
        """
        pass

    def train(self, cell_line_input: FeatureDataset, drug_input: str, output: DrugResponseDataset):
        """
        Trains the model.
        :param cell_line_input: training data associated with the cell line input
        :param drug_input: drug name
        :param output: training data associated with the reponse output
        """
        self.train_drug(cell_line_input, drug_input, output)

    @abstractmethod
    def train_drug(self, cell_line_input: FeatureDataset, drug_name: str, output: DrugResponseDataset):
        """
        Trains one model per drug.
        :param cell_line_input: training data associated with the cell line input
        :param drug_name: drug name
        :param output: training data associated with the reponse output
        """
        pass

    def predict(self, cell_line_input: FeatureDataset, drug_input: str):
        """
        Predicts the response for the given input.
        :param cell_line_input: input associated with the cell line
        :param drug_input: drug name
        :return: predicted response
        """
        self.predict_drug(cell_line_input, drug_input)

    @abstractmethod
    def predict_drug(self, cell_line_input: FeatureDataset, drug_name: str):
        """
        Predicts the response for the given single drug.
        :param cell_line_input: input associated with the cell line
        :param drug_name: drug name
        :return: predicted response
        """
        raise NotImplementedError("predict_drug method not implemented")

    @abstractmethod
    def evaluate(self, cell_line_input: FeatureDataset, drug_input: str, output: DrugResponseDataset):
        """
        Evaluates the model.
        :param cell_line_input: evaluation data associated with the cell line input
        :param drug_input: evaluation data associated with the drug input
        :param output: evaluation data associated with the response output
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Saves the model to models_saved.
        :param path: path to save the model
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Loads the model.
        :param path: path to load the model
        """
        pass

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
