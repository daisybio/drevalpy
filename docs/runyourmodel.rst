Run your own model
===================

DrEvalPy provides a standardized interface for running your own model.
First make a new folder for your model at ``drevalpy/models/your_model_name``.
Create ``drevalpy/models/your_model_name/your_model.py`` in which you need to define the Python class for your model.
This class should inherit from the :ref:`DRPModel <DRP-label>` base class.
DrEvalPy is agnostic to the specific modeling strategy you use. However, you need to define the input views, which represent different types of features that your model requires.
In this example, the model uses "gene_expression" and "methylation" features as cell line views, and "fingerprints" as drug views.
Additionally, you must define a unique model name to identify your model during evaluation.

.. code-block:: Python

    from drevalpy.models.drp_model import DRPModel
    from drevalpy.datasets.dataset import FeatureDataset

    import pandas as pd

    class YourModel(DRPModel):
        """A revolutionary new modeling strategy."""

        is_single_drug_model = True / False # TODO: set to true if your model is a single drug model
        early_stopping = True / False # TODO: set to true if you want to use a part of the validation set for early stopping
        cell_line_views = ["gene_expression", "methylation"]
        drug_views = ["fingerprints"]
        model_name = "YourModel"


Next let's implement the feature loading. You have to return a DrEvalPy FeatureDataset object which contains the features for the cell lines and drugs.
If the features are different depending on the dataset, use the ``dataset_name`` parameter.

.. code-block:: Python

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the drug features, in this case the drug ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the drug ids
        """
        feature_dataset = FeatureDataset.from_csv(f"{data_path}/{dataset_name}/fingerprints.csv") # make sure to adjust the path to your data

        return feature_dataset


    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features, in this case the cell line ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the cell line ids
        """
        feature_dataset = FeatureDataset.from_csv(f"{data_path}/{dataset_name}_gene_expression.csv",
                                                    id_column="cell_line_ids",
                                                    view_name="gene_expression"
                                                 ) # make sure to adjust the path to your data
        methylation = FeatureDataset.from_csv(f"{data_path}/{dataset_name}_methylation.csv",
                                                id_column="cell_line_ids",
                                                view_name="gene_expression"
                                             ) # make sure to adjust the path to your data
        feature_dataset.add_features(methylation)

        return feature_dataset

The build_model functions can be used if you want to use tunable hyperparameters.
The hyperparameters which get tested are defined in the ``drevalpy/models/your_model_name/hyperparameters.yaml``.

.. code-block:: Python

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Builds the model, for models that use hyperparameters.

        :param hyperparameters: hyperparameters for the model
        Example:
            self.model = ElasticNet(alpha=hyperparameters["alpha"], l1_ratio=hyperparameters["l1_ratio"])
        """
        predictor = YourPredictor(hyperparameters) # Initialize your Predictor, this could be a sklearn model, a neural network, etc.

Sometimes, the model design is dependent on your training data input. In this case, you can also consider implementing build_model like:

.. code-block:: Python

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        self.hyperparameters = hyperparameters

and then set the model design later in the train method when you have access to the training data.
The train method should handle model training, and saving any necessary information (e.g., learned parameters).
Here we use a simple predictor that just uses the concatenated features to predict the response.

.. code-block:: Python

    def train(self, output: DrugResponseDataset, cell_line_input: FeatureDataset, drug_input: FeatureDataset | None = None, output_earlystopping: DrugResponseDataset | None = None) -> None:

        inputs = self.get_feature_matrices(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

        predictor.fit(**inputs, output.response)

        self.predictor = predictor # save your predictor for the prediction step

In case you want to set some parameters dependent on the training data, your train function might look like this:

.. code-block:: Python

    def train(self, output: DrugResponseDataset, cell_line_input: FeatureDataset, drug_input: FeatureDataset | None = None, output_earlystopping: DrugResponseDataset | None = None) -> None:

        cell_line_input = self._feature_selection(output, cell_line_input)
        dim_gex, dim_mut, dim_cnv = get_dimensions_of_omics_data(cell_line_input)

        self.nn_model = YourModel(
                                input_size_gex=dim_gex,
                                input_size_mut=dim_mut,
                                input_size_cnv=dim_cnv,
                                hpams=self.hyperparameters,
                                ...
                            )
        self.nn_model.fit(
            output_train=output,
            output_early_stopping=output_earlystopping,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

The predict method should handle model prediction, and return the predicted response values.

.. code-block:: Python

    def predict(self, cell_line_input: FeatureDataset, drug_input: FeatureDataset | None = None) -> np.ndarray:

        inputs = self.get_feature_matrices(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

        return self.predictor.predict(**inputs, output.response)

Finally, you need to register your model with the framework. This can be done by adding the following line to the ``__init__.py`` file in the ``drevalpy/models/__init__.py`` directory.
Update the ``MULTI_DRUG_MODEL_FACTORY`` if your model is a global model for multiple cancer drugs or to the ``SINGLE_DRUG_MODEL_FACTORY`` if your model is specific to a single drug and needs to be trained for each drug separately.

.. code-block:: Python

    from .your_model_name.your_model import YourModel
    MULTI_DRUG_MODEL_FACTORY.update("YourModel": YourModel)


Next, please also write appropriate tests in ``tests/individual_models``.
