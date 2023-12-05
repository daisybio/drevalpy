from abc import ABC, abstractmethod

class Dataset(ABC):
    """
    Abstract wrapper class for datasets.
    """
    def __init__(self, path: str, name: str):
        """
        Initializes the dataset.
        :param path: path to the dataset
        :param name: name of the dataset
        """
        self.path = path
        self.name = name

    @abstractmethod
    def load(self):
        """
        Loads the dataset from data.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Saves the dataset to data.
        """
        pass


class DrugResponseDataset(Dataset):
    """
    Drug response dataset.
    """
    def __init__(self, path: str, name: str, target_type: str, *args, **kwargs):
        """
        Initializes the drug response dataset.
        :param path: path to the dataset
        :param name: name of the dataset
        :param target_type: type of the target value (IC50, EC50, AUC, classification)
        :param args: additional arguments
        :param kwargs: additional keyword arguments

        Variables:
        response: drug response values per cell line and drug
        cell_line_ids: cell line IDs
        drug_ids: drug IDs
        predictions: optional. Predicted drug response values per cell line and drug
        """
        super(DrugResponseDataset, self).__init__(path, name)
        self.target_type = target_type
        self.response, self.cell_line_ids, self.drug_ids = self.read_data(*args, **kwargs)
        self.predictions = None

    def read_data(self, *args, **kwargs):
        """
        Reads the data from the input path with possible additional inputs. Returns the responses, cell line IDs and drug IDs.
        """
        cell_line_ids = []
        drug_ids = []
        responses = []
        with open(self.path, 'r') as f:
            pass
        return responses, cell_line_ids, drug_ids

    def load(self):
        """
        Loads the drug response dataset from data.
        """
        raise NotImplementedError("load method not implemented")

    def save(self):
        """
        Saves the drug response dataset to data.
        """
        raise NotImplementedError("save method not implemented")

    def split_dataset(self, mode):
        """
        Splits the dataset into training, validation and test sets.
        :param mode: split mode (LPO=Leave-random-Pairs-Out, LCO=Leave-Cell-line-Out, LDO=Leave-Drug-Out)
        :return: training, validation and test sets
        """
        raise NotImplementedError("split_dataset method not implemented")


class FeatureDataset(Dataset):
    """
    Abstract wrapper class for feature datasets.
    """
    def __init__(self, path: str, name: str):
        """
        Initializes the feature dataset.
        :param path: path to the dataset
        :param name: name of the dataset
        :param features: features of the dataset
        """
        super(FeatureDataset, self).__init__(path, name)
        self.features = None
        
    @abstractmethod
    def load(self):
        """
        Loads the feature dataset from data.
        """
        pass
    
    @abstractmethod
    def save(self):
        """
        Saves the feature dataset to data.
        """
        pass


class DrugFeatureDataset(FeatureDataset):
    """
    Abstract wrapper class for drug feature datasets.
    """
    def __init__(self, path: str, name: str, *args, **kwargs):
        """
        Initializes the drug feature dataset.
        :param path: path to the dataset
        :param name: name of the dataset

        features: dictionary of features, key: drug ID, value: feature vector
        """
        super(DrugFeatureDataset, self).__init__(path, name)
        self.features, self.drug_ids = self.read_data(*args, **kwargs)

    def read_data(self, *args, **kwargs):
        """
        Reads the data from the input path and possible additional inputs.
        Returns the drug feature vectors and the drug IDs
        """
        drug_ids = []
        features = []
        with open(self.path, 'r') as f:
            pass
        return features, drug_ids

    def load(self):
        """
        Loads the drug feature dataset from data.
        """
        raise NotImplementedError("load method not implemented")

    def save(self):
        """
        Saves the drug feature dataset to data.
        """
        raise NotImplementedError("save method not implemented")

    def randomize_drug_features(self):
        """
        Randomizes the drug feature vectors.
        """
        raise NotImplementedError("randomize_drug_features method not implemented")

    def normalize_drug_features(self, mode: str):
        """
        Normalizes the drug feature vectors.
        """
        raise NotImplementedError("normalize_drug_features method not implemented")


class CellLineFeatureDataset(FeatureDataset):
    """
    Abstract wrapper class for cell line feature datasets.
    """
    def __init__(self, path: str, name: str, *args, **kwargs):
        """
        Initializes the cell line feature dataset.
        :param path: path to the dataset
        :param name: name of the dataset

        features: dictionary of features, key: cell line ID, value: feature vector
        """
        super(CellLineFeatureDataset, self).__init__(path, name)
        self.features, self.cell_line_ids = self.read_data(*args, **kwargs)

    def read_data(self, *args, **kwargs):
        """
        Reads the data from the input path and possible additional inputs.
        Returns the cell line feature vectors and the cell line IDs
        """
        cell_line_ids = []
        features = []
        with open(self.path, 'r') as f:
            pass
        return features, cell_line_ids

    def load(self):
        """
        Loads the cell line feature dataset from data.
        """
        raise NotImplementedError("load method not implemented")

    def save(self):
        """
        Saves the cell line feature dataset to data.
        """
        raise NotImplementedError("save method not implemented")

    def randomize_cell_line_features(self):
        """
        Randomizes the cell line feature vectors.
        """
        raise NotImplementedError("randomize_cell_line_features method not implemented")

    def normalize_cell_line_features(self, mode: str):
        """
        Normalizes the cell line feature vectors.
        """
        raise NotImplementedError("normalize_cell_line_features method not implemented")