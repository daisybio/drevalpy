Port a 1.x model to components
==============================

.. interface: python

A 1.x model was one ``DRPModel`` subclass that loaded its own features, trained
and predicted. In v2 those jobs belong to separate components: a sample
featurizer, a drug featurizer and a predictor. A model is a recipe that names
them. This page ports one 1.x class step by step. :doc:`/explanation/models_from_components`
explains why the work is split this way.

The 1.x model to port
---------------------

This is the model from the 1.x guide "Implement your model", with the imports
kept. It uses two omics tables for the samples and the drug fingerprints, and
fits a ridge regression on the concatenated features.

.. snippet: skip

.. code-block:: python

   from typing import Any

   import numpy as np
   from sklearn.linear_model import Ridge

   from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
   from drevalpy.models.drp_model import DRPModel


   class YourModel(DRPModel):
       is_single_drug_model = False
       early_stopping = False
       cell_line_views = ["gene_expression", "methylation"]
       drug_views = ["fingerprints"]

       @classmethod
       def get_model_name(cls) -> str:
           return "YourModel"

       def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
           return FeatureDataset.from_csv(...)

       def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
           return FeatureDataset.from_csv(...)

       def build_model(self, hyperparameters: dict[str, Any]) -> None:
           self.predictor = Ridge(alpha=hyperparameters["alpha"])

       def train(self, output: DrugResponseDataset, cell_line_input, drug_input=None,
                 output_earlystopping=None, model_checkpoint_dir="checkpoints") -> None:
           inputs = self.get_feature_matrices(
               cell_line_ids=output.cell_line_ids,
               drug_ids=output.drug_ids,
               cell_line_input=cell_line_input,
               drug_input=drug_input,
           )
           self.predictor.fit(**inputs, y=output.response)

       def predict(self, cell_line_ids, drug_ids, cell_line_input, drug_input=None) -> np.ndarray:
           inputs = self.get_feature_matrices(cell_line_ids, drug_ids, cell_line_input, drug_input)
           return self.predictor.predict(**inputs)

It was registered by importing the class in ``drevalpy/models/__init__.py`` and
adding it to ``MULTI_DRUG_MODEL_FACTORY``, and its search space sat in
``hyperparameters.yaml`` beside it.

Where each part goes
--------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - In the 1.x class
     - In v2
   * - ``cell_line_views = [...]``
     - The sample featurizer reads the views. Name the view in the recipe:
       ``raw[gene_expression]+raw[methylation]``. A featurizer declares the
       views it needs in ``input_views``.
   * - ``drug_views = ["fingerprints"]``
     - The drug featurizer, for example ``fingerprints``.
   * - ``load_cell_line_features``, ``load_drug_features``,
       ``FeatureDataset.from_csv``
     - Removed. The features are tables in the dataset ``.h5mu`` file, so the
       featurizer needs no loader. To add your own features, see
       :doc:`/how-to/bring_your_own_data`.
   * - Scaling, feature selection or PCA done in ``train`` with
       ``scale_gene_expression`` and similar helpers
     - A featurizer, fitted on the training samples only so that nothing leaks.
       Shipped ones include ``scaledGeneExpression``, ``pca`` and ``raw``.
       Write your own as :doc:`/how-to/extend/write_featurizer` describes.
   * - ``build_model``, ``train``, ``predict``
     - A predictor: ``_fit_matrix`` and ``_predict_matrix`` for a
       ``MatrixPredictor``. The base class turns the pairs into the design
       matrix, so you no longer call ``get_feature_matrices``.
   * - ``hyperparameters.yaml`` and ``get_hyperparameter_set``
     - ``get_hyperparameter_space`` on the predictor: one entry per
       hyperparameter with a type, a range and a ``default``. Optuna samples
       from it. Nothing reads a YAML grid.
   * - ``get_model_name``
     - The name of the registration, and the name of the ``ModelSpec``.
   * - ``is_single_drug_model = True``
     - A predictor that fits one model per drug, such as the shipped
       ``singleDrugElasticNet``. See :doc:`/how-to/extend/write_predictor`.
   * - ``early_stopping = True``
     - The class attributes ``supports_early_stopping`` or
       ``requires_early_stopping`` on the predictor.
   * - ``save`` and ``load`` of your own
     - ``get_state`` and ``set_state`` return and restore the fitted state. The
       checkpoint format is the same for every model.
   * - ``MODEL_FACTORY``, ``MULTI_DRUG_MODEL_FACTORY``,
       ``SINGLE_DRUG_MODEL_FACTORY`` entry
     - The decorator ``register_predictor`` or ``register_sample_featurizer``.
       No file in the package is edited.
   * - ``drevalpy --models YourModel``
     - ``drevalpy -e DIR run YourModel ...``, after the predictor and a zoo
       preset named ``YourModel`` are in a directory that ``-e`` loads. See
       :doc:`/how-to/extend/load_extensions`. ``run`` takes zoo preset names,
       not recipes.

Write the predictor
-------------------

The predictor below is the ridge regression of the 1.x class. It is registered
under the name ``migratedRidge``, which must be unique.

.. code-block:: python

   from typing import Any

   import numpy as np
   from sklearn.linear_model import Ridge

   from drevalpy.plugin import FeatureFormat, MatrixPredictor, register_predictor


   @register_predictor(
       "migratedRidge",
       description="Ridge regression on concatenated dense sample and drug features.",
       sample_contract=FeatureFormat.NUMERIC_MATRIX,
       drug_contract=FeatureFormat.NUMERIC_MATRIX,
   )
   class MigratedRidge(MatrixPredictor):
       def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
           super().__init__(hyperparameters)
           self._estimator: Ridge | None = None

       @classmethod
       def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
           return {"alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0}}

       def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
           self._estimator = Ridge(alpha=float(self._hyperparameters["alpha"]))
           self._estimator.fit(x, y)

       def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
           return np.asarray(self._estimator.predict(x), dtype=np.float64)

       def get_state(self) -> dict[str, object]:
           if self._estimator is None:
               return {}
           return {"estimator": self._estimator, "hyperparameters": dict(self._hyperparameters)}

       def set_state(self, state: dict[str, object]) -> None:
           self._estimator = state.get("estimator")
           self._hyperparameters = dict(state.get("hyperparameters", self._hyperparameters))

       def is_fitted(self) -> bool:
           return self._estimator is not None

The shipped example ``docs/examples/toy_ridge_predictor.py`` is the same
predictor with docstrings, and :doc:`/how-to/extend/write_predictor` explains
every hook.

Name the model and run it
-------------------------

A model is a ``ModelSpec``: a name and a recipe. The recipe
``sample featurizer:drug featurizer:predictor`` replaces the two view lists of
the 1.x class. For the gene expression, methylation and fingerprints of the 1.x
model it reads ``raw[gene_expression]+raw[methylation]:fingerprints:migratedRidge``.
That needs a dataset with those tables.

The synthetic dataset has no omics, so this run uses the featurizers that need
none: ``tissue`` one-hot encodes the tissue of each sample and ``identity``
one-hot encodes each drug.

.. code-block:: python

   import drevalpy
   from drevalpy import ModelSpec
   from drevalpy.models.config import from_spec

   dataset = drevalpy.synthetic(n_samples=60, n_drugs=8, seed=3)
   spec = ModelSpec("MigratedRidge", from_spec("tissue:identity:migratedRidge"))
   baseline = ModelSpec.from_name("NaiveMeanEffectsPredictor")

   result = drevalpy.run([spec, baseline], dataset, "LSO", hyperparameter_tuning=False)
   print(result.summary_table())

Tuning now searches the space your predictor declares:

.. code-block:: python

   tuned = drevalpy.run([spec], dataset, "LSO", hpo_num_trials=3)
   print(tuned.models[0].runs[0].best_hyperparameters)

The registration lives in the process that ran it. To use the model from the
command line, put the predictor in an extension file, add a zoo file that names
the recipe, and load both with ``drevalpy -e DIR``
(:doc:`/reference/extension_files`). The predictor then shows up in
``drevalpy list predictors`` and the preset in ``drevalpy list zoo-presets``.

Check the port
--------------

Numbers from a ported model need not match the 1.x model, even when the
algorithm is the same. Preprocessing, the choice of folds, the response scale
(pEC50 in v2) and the search over hyperparameters have all changed. Compare both
against a baseline on the same data instead of comparing the two outputs. To
test the predictor itself, use the shipped conformance checks, as
:doc:`/how-to/extend/test_components` shows.

What does not carry over
------------------------

- A 1.x checkpoint does not load into the ported model, and neither do 1.x
  predictions or splits. Retrain and re-run. The table is at
  :ref:`how-to/migrate/index:Which 1.x files still load`.
- The model names that do exist are listed in
  :ref:`how-to/migrate/index:Map 1.x model names to v2`.
