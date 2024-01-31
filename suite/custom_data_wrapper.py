from suite.data_wrapper import DrugResponseDataset


class CustomDRPDataset(DrugResponseDataset):
    """
    Wrapper class to be implemented for a new custom drug response dataset.
    """

    def __init__(self, path: str, name: str, target_type: str, *args, **kwargs):
        """
        Initializes the dataset wrapper.
        :param path: path to the dataset
        :param name: name of the dataset
        :param target_type: type of the target value
        """
        super(CustomDRPDataset, self).__init__(path, name, target_type, *args, **kwargs)

    def read_data(self, *args, **kwargs):
        """
        Reads the data from the input path with possible additional inputs. Returns the responses, cell line IDs and drug IDs.
        """
        cell_line_ids = []
        drug_ids = []
        responses = []
        with open(self.path, "r") as f:
            pass
        return responses, cell_line_ids, drug_ids

    def load(self):
        """
        Loads the drug response dataset.
        """
        raise NotImplementedError("load method not implemented")

    def save(self):
        """
        Saves the drug response dataset.
        """
        raise NotImplementedError("save method not implemented")

    def split_dataset(self, mode):
        """
        Splits the dataset into training, validation and test sets.
        :param mode: split mode (random, cell_line, drug)
        :return: training, validation and test sets
        """
        raise NotImplementedError("split_dataset method not implemented")
