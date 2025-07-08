Run your own model
===================

DrEvalPy provides a standardized interface for running your own model.

There are a few steps to follow so we can make sure the model evaluation is consistent and reproducible.
Feel free to contact us via GitHub if you experience any difficulties :-)

First, make a new folder for your model at ``drevalpy/models/your_model_name``.
Create ``drevalpy/models/your_model_name/your_model.py``, in which you need to define the Python class for your model.
This class should inherit from the :ref:`DRPModel <DRP-label>` base class.
DrEvalPy is agnostic to the specific modeling strategy you use. However, you need to define the input views (a.k.a modalities), which represent different types of features that your model requires.
In this example, the model uses "gene_expression" and "methylation" features as cell line views, and "fingerprints" as drug views.
Additionally, you must define a unique model name to identify your model during evaluation.

.. code-block:: Python

    from drevalpy.models.drp_model import DRPModel
    from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset
    from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER
    from drevalpy.models.utils import (
        load_and_select_gene_features,
        load_drug_fingerprint_features,
        scale_gene_expression,
    )
    from typing import Any
    import numpy as np

    class YourModel(DRPModel):
        """A revolutionary new modeling strategy."""

        is_single_drug_model = True / False # TODO: set to true if your model is a single drug model (i.e., it needs to be trained for each drug separately)
        early_stopping = True / False # TODO: set to true if you want to use a part of the validation set for early stopping
        cell_line_views = ["gene_expression", "methylation"]
        drug_views = ["fingerprints"]

        @classmethod
        def get_model_name(cls) -> str:
            """
            Returns the name of the model.
            """
            return "YourModel"

Next let's implement the feature loading. You have to return a DrEvalPy FeatureDataset object which contains the features for the cell lines and drugs.
If the features are different depending on the dataset, use the ``dataset_name`` parameter.
In this example, we load custom drug fingerprints and cell line gene expression and methylation features from CSV files.
The cell line ids of your gene expression and methylation csvs should match the ``CELL_LINE_IDENTIFIER`` ("cell_line_name"),
and the drug ids of your fingerprints csv should match the ``DRUG_IDENTIFIER`` ("pubchem_id").
The model will use these identifiers to match the features with the drug response data.

:download:`Example fingerprint file <_static/example_data/fingerprints_example.csv>`, :download:`Example gene expression file <_static/example_data/gex_example.csv>`.

For our provided datasets, we have other loading methods implemented in the `drevalpy/models/utils.py` file, which you can also use:

* ``def load_and_select_gene_features``: Loads a specified omic; enables selecting a specific gene list (e.g., landmark genes).
* ``def get_multiomics_feature_dataset``: Loads the specified omics (iteratively calls previous method).
* ``def load_drug_fingerprint_features``: Loads the provided drug fingerprints.
* ``def load_cl_ids_from_csv``
* ``def load_drug_ids_from_csv``
* ``load_tissues_from_csv``

.. code-block:: Python

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the drug features, in this case the drug ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the drug ids
        """
        feature_dataset = FeatureDataset.from_csv(
            path_to_csv=f"{data_path}/{dataset_name}/fingerprints.csv",
            id_column=DRUG_IDENTIFIER,
            view_name="fingerprints",
            drop_columns=None
        ) # make sure to adjust the path to your data. If you want to drop columns, specify them in a list.

        return feature_dataset


    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features, in this case the cell line ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the cell line ids
        """
        feature_dataset = FeatureDataset.from_csv(f"{data_path}/{dataset_name}_gene_expression.csv",
                                                    id_column=CELL_LINE_IDENTIFIER,
                                                    view_name="gene_expression",
                                                    drop_columns=['cellosaurus_id']
                                                 ) # make sure to adjust the path to your data
        methylation = FeatureDataset.from_csv(f"{data_path}/{dataset_name}_methylation.csv",
                                                id_column=CELL_LINE_IDENTIFIER,
                                                view_name="methylation",
                                                drop_columns=['cellosaurus_id']
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
        self.predictor = YourPredictor(hyperparameters) # Initialize your Predictor, this could be a sklearn model, a neural network, etc.

Sometimes, the model design is dependent on your training data input. In this case, you can also consider implementing build_model like:

.. code-block:: Python

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        self.hyperparameters = hyperparameters

and then set the model design later in the train method when you have access to the training data.
(e.g., when you can access the feature dimensionalities)
The train method should handle model training, and saving any necessary information (e.g., learned parameters).
Here we use a simple predictor that just uses the concatenated features to predict the response.

.. code-block:: Python

    def train(self, output: DrugResponseDataset, cell_line_input: FeatureDataset, drug_input: FeatureDataset | None = None, output_earlystopping: DrugResponseDataset | None = None, model_checkpoint_dir: str | None = None) -> None:

        inputs = self.get_feature_matrices(
            cell_line_ids=output.cell_line_ids,
            drug_ids=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

        self.predictor.fit(**inputs, output.response)

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


We also provide utility functions (drevalpy/models/utils.py) for data transformations that have to be computed on the training data only (e.g., scaling, feature selection) to avoid data leakage:

* ``def scale_gene_expression``
* ``class VarianceFeatureSelector``
* ``def prepare_expression_and_methylation``
* ``class ProteomicsMedianCenterAndImputeTransformer``
* ``def prepare_proteomics``


The predict method should handle model prediction, and return the predicted response values.

.. code-block:: Python

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.

        :param drug_ids: list of drug ids, also used for single drug models, there it is just an array containing the
            same drug id
        :param cell_line_ids: list of cell line ids
        :param cell_line_input: input associated with the cell line, required for all models
        :param drug_input: input associated with the drug, optional because single drug models do not use drug features
        :returns: predicted response
        """

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

Now you can run your model using the DrEvalPy pipeline. Run the following command (after installing your cloned and edited DrEvalPy repository e.g. with ``pip install -e .``):

.. code-block:: shell
    drevalpy --model YourModel --dataset CTRPv2 --data_path data


To contribute the model, so that the community can build on it, please also write appropriate tests in ``tests/models`` and documentation in ``docs/``
We are happy to help you with that, contact us via GitHub!

Let's look at an example an example implementation of a model using the DrEvalPy framework:



Example: TinyNN (Neural Network with PyTorch)
---------------------------------------------

In this example, we implement a simple feedforward neural network for drug response prediction using gene expression and drug fingerprint features.
We use and recommend PyTorch, but you can use any other framework like TensorFlow, JAX, etc.
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
    from drevalpy.models.utils import load_and_select_gene_features, load_drug_fingerprint_features


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

3. We define how the features are loaded. Here, we use our presupplied datasets and the preimplemented functions. Loading features can be customized (Have a look at the FeatureDataset class for more details e.g. on how to load features from a CSV).

.. code-block:: Python

        def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:

            return load_and_select_gene_features(feature_type="gene_expression",
                                                data_path=data_path,
                                                dataset_name=dataset_name,
                                                gene_list="landmark_genes")


        def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:

            return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)

1. In the ``build_model`` we just store the hyperparameters.

.. code-block:: Python

        def build_model(self, hyperparameters: dict[str, Any]) -> None:
            self.hyperparameters = hyperparameters

5. In the train method we scale gene expression and train the model.

.. code-block:: Python

        def train(self, output, cell_line_input, drug_input, output_earlystopping=None, model_checkpoint_dir=None):
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

7. Add hyperparameters to your ``hyperparameters.yaml``. We add two values for the hidden layer size. DrEval will tune over this hyperparameter space.

.. code-block:: YAML

    TinyNN:
      hidden_dim:
        - 32
        - 64

8. Register the model in ``models/__init__.py``.

.. code-block:: Python

    from .your_model_folder.tinynn import TinyNN
    MULTI_DRUG_MODEL_FACTORY.update({"TinyNN": TinyNN})



Second Example: ProteomicsRandomForest
--------------------------------------

Instead of gene expression data, we want to use proteomics data in our Random Forest.
The Random Forest model is already implemented in ``models/baselines/sklearn_models.py``.
We now adapt it to work with proteomics features, and apply preprocessing steps including missing value imputation, feature selection, and normalization.

1. We create a new class ``ProteomicsRandomForest`` which inherits from ``RandomForest``.
We overwrite ``cell_line_views`` to ``["proteomics"]`` and define the model name.

.. code-block:: python
    from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
    from drevalpy.models import RandomForest
    from models.utils import (
        ProteomicsMedianCenterAndImputeTransformer,
        load_and_select_gene_features,
        load_drug_fingerprint_features,
        prepare_proteomics,
        scale_gene_expression,
    )

    class ProteomicsRandomForest(RandomForest):
        """RandomForest model for drug response prediction using proteomics data."""

        cell_line_views = ["proteomics"]

        def __init__(self):
            super().__init__()
            self.feature_threshold = 0.7
            self.n_features = 1000
            self.normalization_width = 0.3
            self.normalization_downshift = 1.8

        @classmethod
        def get_model_name(cls) -> str:
            return "ProteomicsRandomForest"

1. We implement the ``build_model`` method to configure the preprocessing transformer from hyperparameters.

.. code-block:: python

        def build_model(self, hyperparameters: dict) -> None:
            super().build_model(hyperparameters)
            self.feature_threshold = hyperparameters.get("feature_threshold", 0.7)
            self.n_features = hyperparameters.get("n_features", 1000)
            self.normalization_width = hyperparameters.get("normalization_width", 0.3)
            self.normalization_downshift = hyperparameters.get("normalization_downshift", 1.8)
            self.proteomics_transformer = ProteomicsMedianCenterAndImputeTransformer(
                feature_threshold=self.feature_threshold,
                n_features=self.n_features,
                normalization_downshift=self.normalization_downshift,
                normalization_width=self.normalization_width,
            )

3. We implement the ``load_cell_line_features`` method to load the presupplied proteomics features.

.. code-block:: python

        def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
            return load_and_select_gene_features(
                feature_type="proteomics",
                gene_list=None,
                data_path=data_path,
                dataset_name=dataset_name,
            )

4. We implement the ``train`` method and preprocess the features before training.

.. code-block:: python

        def train(
            self,
            output: DrugResponseDataset,
            cell_line_input: FeatureDataset,
            drug_input: FeatureDataset | None = None,
            output_earlystopping: DrugResponseDataset | None = None,
            model_checkpoint_dir: str = "checkpoints",
        ) -> None:
            if drug_input is None:
                raise ValueError("drug_input (fingerprints) is required.")
            cell_line_input = prepare_proteomics(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(output.cell_line_ids),
                training=True,
                transformer=self.proteomics_transformer,
            )
            x = self.get_concatenated_features(
                cell_line_view=self.cell_line_views[0],
                drug_view=self.drug_views[0],
                cell_line_ids_output=output.cell_line_ids,
                drug_ids_output=output.drug_ids,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
            )
            self.model.fit(x, output.response)

5. We implement the ``predict`` method and apply the same preprocessing.

.. code-block:: python

        def predict(
            self,
            cell_line_ids: np.ndarray,
            drug_ids: np.ndarray,
            cell_line_input: FeatureDataset,
            drug_input: FeatureDataset | None = None,
        ) -> np.ndarray:
            if drug_input is None:
                raise ValueError("drug_input (fingerprints) is required.")
            cell_line_input = prepare_proteomics(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(cell_line_ids),
                training=False,
                transformer=self.proteomics_transformer,
            )
            if self.model is None:
                return np.full(len(cell_line_ids), np.nan)
            x = self.get_concatenated_features(
                cell_line_view=self.cell_line_views[0],
                drug_view=self.drug_views[0],
                cell_line_ids_output=cell_line_ids,
                drug_ids_output=drug_ids,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
            )
            return self.model.predict(x)

6. We define the hyperparameters in ``models/baselines/hyperparameters.yaml``.

.. code-block:: yaml

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
      feature_threshold:
        - 0.7
      n_features:
        - 1000
      normalization_width:
        - 0.3
      normalization_downshift:
        - 1.8

7. We register the model in ``models/__init__.py``.

.. code-block:: python

    from .baselines.sklearn_models import ProteomicsRandomForest

    MULTI_DRUG_MODEL_FACTORY.update({
        "ProteomicsRandomForest": ProteomicsRandomForest,
    })


Now you can run the model using the DrEvalPy pipeline.
To run the model and execute the following command:
.. code-block:: shell

    drevalpy --model ProteomicsRandomForest --dataset CTRPv2 --data_path data