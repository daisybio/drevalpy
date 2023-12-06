from suite.model_wrapper import SingleDRPModel
from suite.data_wrapper import DrugResponseDataset, FeatureDataset


class CustomSingleDrugModelWrapper(SingleDRPModel):
    """
    Wrapper class to be implemented for a new custom model that builds one model per drug.
    """
    def build_model(self, *args, **kwargs):
        pass

    def train_drug(self, cell_line_input: FeatureDataset, drug_name: str, output: DrugResponseDataset):
        pass

    def predict_drug(self, cell_line_input: FeatureDataset, drug_name: str):
        pass

    def evaluate(self, cell_line_input: FeatureDataset, drug_input: str, output: DrugResponseDataset):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def get_cell_line_features(self, cell_line_input: FeatureDataset):
        pass

    def get_drug_features(self, drug_input: FeatureDataset):
        pass
