Run your own model
====

DrEvalPy provides an easy-to-use interface for running your own model. First, you need to define a Python class for your model. This class should inherit from the DRPModel base class.
The framework is agnostic to the specific modeling strategy you use.
However, you need to define the input views, which represent different types of features that your model requires.
In this example, the model uses "gene_expression" and "methylation" features as cell line views, and "fingerprints" as drug views.
Additionally, you must define a unique model name to identify your model during evaluation.

.. code-block:: Python

    from drevalpy.models.drp_model import DRPModel
    from drevalpy.datasets.dataset import FeatureDataset

    import pandas as pd

    class YourModel(DRPModel):
        """A revolutionary new modeling strategy."""

        cell_line_views = ["gene_expression", "methylation"]
        drug_views = ["fingerprints"]
        model_name = "YourModel"


Next let's implement the feature loading. You have to return a DrEvalPy FeatureDataset object which contains the features for the cell lines and drugs.
If the features are different depending on the dataset, use the `dataset_name` parameter.

.. code-block:: Python

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the drug features, in this case the drug ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the drug ids
        """
        feature_dataset = FeatureDataset.from_csv(f"{data_path}/{dataset_name}_fingerprints.csv")

        return feature_dataset


    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features, in this case the cell line ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the cell line ids
        """
        feature_dataset = FeatureDataset.from_csv(f"{data_path}/{dataset_name}_gene_expression.csv")
        methylation = FeatureDataset.from_csv(f"{data_path}/{dataset_name}_methylation.csv")
        feature_dataset.add_features(methylation)

        return feature_dataset

The build_model functions can be used if you want to use tunable hyperparameters. The hyperparameters which get tested are defined in the hyperparameters.yaml which is in the same folder as your model class.

def build_model(self, hyperparameters: dict[str, Any]) -> None:
    self.hyperparameters = hyperparameters


The train method should handle model training, and saving any necessary information (e.g., learned parameters). Here we use a simple predictor that just uses the concatenated features to predict the response. (TODO add Methylation)

.. code-block:: Python

    def train(self, output: DrugResponseDataset, cell_line_input: FeatureDataset, drug_input: FeatureDataset | None = None) -> None:

        predictor = YourPredictor(self.hyperparameters) # Initialize your Predictor

        # Example using sklearn's fit method
        X = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input
        )
        predictor.fit(X, output.response)

        self.predictor = predictor # save your predictor for the prediciton step

The predict method should handle model prediction, and return the predicted response values.

.. code-block:: Python

    def predict(self, cell_line_input: FeatureDataset, drug_input: FeatureDataset | None = None) -> np.ndarray:

        X = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_input=cell_line_input,
            drug_input=drug_input
        )

        return self.predictor.predict(X)

Finally, you need to register your model with the framework. This can be done by adding the following line to the `__init__.py` file in the `models` directory.



