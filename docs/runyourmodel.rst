Run your own model
===================

DrEvalPy provides a standardized interface for running your own model.
First make a new folder for your model at ``drevalpy/models/your_model_name``.
Create ``drevalpy/models/your_model_name/your_model.py`` in which you need to define the Python class for your model.
This class should inherit from the :ref:`DRPModel <DRP-label>` base class.
DrEvalPy is agnostic to the specific modeling strategy you use. However, you need to define the input views(a.k.a modalities), which represent different types of features that your model requires.
In this example, the model uses "gene_expression" and "methylation" features as cell line views, and "fingerprints" as drug views.
Additionally, you must define a unique model name to identify your model during evaluation.

.. code-block:: Python

    from drevalpy.models.drp_model import DRPModel
    from drevalpy.datasets.dataset import FeatureDataset

    import pandas as pd

    class YourModel(DRPModel):
        """A revolutionary new modeling strategy."""

        is_single_drug_model = True / False # TODO: set to true if your model is a single drug model (i.e. it needs to be trained for each drug separately)
        early_stopping = True / False # TODO: set to true if you want to use a part of the validation set for early stopping
        cell_line_views = ["gene_expression", "methylation"]
        drug_views = ["fingerprints"]

        def get_model_name(cls) -> str:
            """
            Returns the name of the model.
            """
            return "YourModel"

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
(i.e. when you can access the feature dimensionalities)
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

Now you can run your model using the DrEvalPy pipeline. cd to the drevalpy root directory and run the following command:

.. code-block:: shell
    python -m run_suite.py --model YourModel --dataset CTRPv2 --data_path data


To contribute the model, so that the community can build on it, please also write appropriate tests in ``tests/individual_models`` and documentation in ``docs/``
We are happy to help you with that, contact us via GitHub!

Let's look at an example of how to implement a model using the DrEvalPy framework:



Example: TinyNN (Neural Network with PyTorch)
---------------------------------------------

In this example, we implement a simple feedforward neural network for drug response prediction using gene expression and drug fingerprint features.
Gene expression features are standardized using a ``StandardScaler``, while fingerprint features are used as-is.

1. We define a minimal PyTorch model with CPU/GPU support.

.. code-block:: Python

    import torch
    import torch.nn as nn
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class FeedForwardNetwork(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.to(device)

        def fit(self, x: np.ndarray, y: np.ndarray, lr: float = 1e-3, epochs: int = 100):
            self.train()
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            for _ in range(epochs):
                optimizer.zero_grad()
                loss = loss_fn(self(x_tensor), y_tensor)
                loss.backward()
                optimizer.step()

        def forward(self, x):
            return self.net(x)

        def predict(self, x: np.ndarray) -> np.ndarray:
            self.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
                preds = self(x_tensor).squeeze(1)
                return preds.cpu().numpy()

2. We create the ``TinyNN`` model class that inherits from ``DRPModel``.

.. code-block:: Python

    from drevalpy.models.drp_model import DRPModel
    from drevalpy.datasets.dataset import FeatureDataset
    from sklearn.preprocessing import StandardScaler

    class TinyNN(DRPModel):
        cell_line_views = ["gene_expression"]
        drug_views = ["fingerprints"]
        early_stopping = True

        def __init__(self):
            super().__init__()
            self.model = None
            self.hyperparameters = None
            self.scaler_gex = StandardScaler()

        @classmethod
        def get_model_name(cls) -> str:
            return "TinyNN"

3. We define how the features are loaded.

.. code-block:: Python

        def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
            return FeatureDataset.from_csv(
                f"{data_path}/{dataset_name}/gene_expression.csv",
                id_column="cell_line_ids",
                view_name="gene_expression"
            )

        def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
            return FeatureDataset.from_csv(
                f"{data_path}/{dataset_name}/fingerprints.csv",
                id_column="drug_ids",
                view_name="fingerprints"
            )

4. We store hyperparameters in ``build_model``.

.. code-block:: Python

        def build_model(self, hyperparameters: dict[str, Any]) -> None:
            self.hyperparameters = hyperparameters

1. In the train method we scale gene expression and train the model.

.. code-block:: Python

        def train(self, output, cell_line_input, drug_input, output_earlystopping=None):
            gex = cell_line_input.get_feature_matrix("gene_expression", output.cell_line_ids)
            fp = drug_input.get_feature_matrix("fingerprints", output.drug_ids)

            gex = self.scaler_gex.fit_transform(gex)
            x = np.concatenate([gex, fp], axis=1)
            y = output.response

            self.model = FeedForwardNetwork(
                input_dim=x.shape[1],
                hidden_dim=self.hyperparameters["hidden_dim"]
            )
            self.model.fit(x, y)

6. We apply scaling in ``predict`` and return model outputs.

.. code-block:: Python

        def predict(self, cell_line_ids, drug_ids, cell_line_input, drug_input):
            gex = cell_line_input.get_feature_matrix("gene_expression", cell_line_ids)
            fp = drug_input.get_feature_matrix("fingerprints", drug_ids)

            gex = self.scaler_gex.transform(gex)
            x = np.concatenate([gex, fp], axis=1)

            return self.model.predict(x)

7. Add hyperparameters to your ``hyperparameters.yaml``.

.. code-block:: YAML

    TinyNN:
      hidden_dim:
        - 32
        - 64

8. Register the model in ``models/__init__.py``.

.. code-block:: Python

    from .your_model_folder.tinynn import TinyNN
    MULTI_DRUG_MODEL_FACTORY.update({"TinyNN": TinyNN})


Example: Proteomics Random Forest
---------------------------------
Instead of gene expression data, we want to use proteomics data in our Random Forest.
The Random Forest is already implemented in ``models/baselines/sklearn_models.py``.
We just need to adapt some methods.

1. We make a new class ProteomicsRandomForest which inherits from the RandomForest class.
We overwrite ``cell_line_views`` to ``["proteomics"]`` and ``get_model_name`` to ``"ProteomicsRandomForest"``.

.. code-block:: Python

    class ProteomicsRandomForest(RandomForest):
        """RandomForest model for drug response prediction using proteomics data."""

        cell_line_views = ["proteomics"]

        @classmethod
        def get_model_name(cls) -> str:
            """
            Returns the model name.

            :returns: ProteomicsRandomForest
            """
            return "ProteomicsRandomForest"


2. Next, we need to implement the ``load_cell_line_features`` method to load the proteomics features.
We already supply proteomics features in the Zenodo as proteomics.csv. Hence, we can already use our
pre-implemented method ``load_cell_line_features``.

.. code-block:: Python

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: Path to the gene expression and landmark genes
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the cell line proteomics features, filtered through the landmark genes
        """
        return load_and_select_gene_features(
            feature_type="proteomics",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name,
        )


3. We use the same build_model method as the RandomForest class, so we don't need to implement it.
However, we need to write the hyperparameters needed into the ``models/baselines/hyperparameters.yaml`` file:

.. code-block:: YAML

    ProteomicsRandomForest:
      n_estimators:
        - 100
      max_depth:
        - 5
        - 10
        - 30
      max_samples:
        - 0.2
      n_jobs:
        - -1
      criterion:
        - squared_error

We also use the same train and predict method as the RandomForest class, so we don't need to implement them.

4. Finally, we need to register the model in the ``__init__.py`` file in the ``models/baselines`` directory.

.. code-block:: Python

    __all__ = [
        "MULTI_DRUG_MODEL_FACTORY",
        "SINGLE_DRUG_MODEL_FACTORY",
        "MODEL_FACTORY",
        "NaivePredictor",
        #[...]
        "DIPKModel",
        "ProteomicsRandomForest"
    ]
    #[...]
    from .baselines.sklearn_models import (
        ElasticNetModel, GradientBoosting, RandomForest, SVMRegressor, ProteomicsRandomForest
    )

    # SINGLE_DRUG_MODEL_FACTORY is used in the pipeline!
    SINGLE_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]] = {
        #[...]
    }

    # MULTI_DRUG_MODEL_FACTORY is used in the pipeline!
    MULTI_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]] = {
        "NaivePredictor": NaivePredictor,
        #[...]
        "DIPK": DIPKModel,
        "ProteomicsRandomForest": ProteomicsRandomForest,
    }


5. Add your model to the tests, in this case in ``tests/individual_models/test_baselines.py``.

.. code-block:: Python

    @pytest.mark.parametrize(
        "model_name",
        [
            "NaivePredictor",
            "NaiveDrugMeanPredictor",
            "NaiveCellLineMeanPredictor",
            "NaiveMeanEffectsPredictor",
            "ElasticNet",
            "RandomForest",
            "SVR",
            "MultiOmicsRandomForest",
            "GradientBoosting",
            "ProteomicsRandomForest",
        ],
    )
    @pytest.mark.parametrize("test_mode", ["LPO", "LCO", "LDO"])
    def test_baselines(
        sample_dataset: DrugResponseDataset,
        model_name: str,
        test_mode: str,
        cross_study_dataset: DrugResponseDataset,
    ) -> None:
    # [...]

6. Now, we add the appropriate documentation.
In our case, the class methods etc. under API are rendered automatically because it is a subclass of Sklearn Models.
If you implement a new model, please orient yourself on the documentation of, e.g., DIPK.

Add your model in ``usage.rst`` under the section "Available models".
