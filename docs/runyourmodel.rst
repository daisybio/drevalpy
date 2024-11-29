Run your own model
====

DrEvalPy provides an easy-to-use interface for running your own model. The framework is agnostic to the specific modeling strategy you use. However, you need to define the input views, which represent different types of features that your model requires. In this example, the model uses "gene_expression" and "methylation" features as cell line views, and "fingerprints" as drug views. Additionally, you must define a unique model name to identify your model during evaluation.

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




The train method should handle model training, and saving any necessary information (e.g., learned parameters).

.. code-block:: Python

