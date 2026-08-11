Extensions
==========

If you are reading this, we assume you are already familiar with:

- :doc:`models` — ``construct_model``, recipes, ``ModelConfig``, and lifecycle
- :doc:`/concepts/component_catalog`
- :doc:`/concepts/from_components_to_models`
- :doc:`/concepts/registries`

Every extensible concept in DrEvalPy — predictors, featurizers, splitters,
datasets, and visualizations — is managed by a registry that maps
human-readable names to implementations. This page shows how to register
custom implementations for each extension point and make them available to the
pipeline.

Custom featurizers
------------------

Subclass ``CellLineFeaturizer`` or ``DrugFeaturizer`` and register with
``@drevalpy.registry.cell_line_featurizer.register`` or
``@drevalpy.registry.drug_featurizer.register``.

Declare a ``FeatureFormat`` **contract** on registration
(``numeric_matrix``, ``graph``, or ``ragged_sequence``). That is the payload
format this featurizer produces. Composition validation compares it to the
predictor's ``cell_line_contract`` / ``drug_contract`` and rejects stacks
where the formats disagree. Registry names and the format vocabulary are listed
in :doc:`/concepts/component_catalog`.

.. code-block:: python

   from __future__ import annotations

   import numpy as np

   from drevalpy.components.core.contracts.contracts import FeatureFormat
   from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
   from drevalpy.registry.cell_line_featurizer import register as register_cell_line_featurizer


   @register_cell_line_featurizer(
       "toyCellLine",
       description="Constant cell-line features for demos.",
       contract=FeatureFormat.NUMERIC_MATRIX,
   )
   class ToyCellLineFeaturizer(CellLineFeaturizer):
       def fit(self, features, *, entity_ids=None):
           self._output_dim = 1
           return self

       def transform(self, features, entity_ids):
           return np.ones((len(entity_ids), 1), dtype=np.float32)

       @property
       def output_dim(self):
           return self._output_dim

       def get_state(self) -> dict[str, object]:
           return {"output_dim": self._output_dim}

       def set_state(self, state: dict[str, object]) -> None:
           output_dim = state.get("output_dim")
           if isinstance(output_dim, int):
               self._output_dim = output_dim

Fitted featurizers must implement ``get_state`` / ``set_state`` so ``*.zip``
checkpoints round-trip.

Custom predictors
-----------------

Every predictor must inherit exactly one input interface and register with
``@drevalpy.registry.predictor.register``. The available types were already
introduced in :doc:`/concepts/component_catalog`.

.. tab-set::

   .. tab-item:: Feature-free

      ``FeatureFreePredictor`` uses pair identifiers and/or response values
      only. Composition forbids cell-line and drug featurizers for it, since it
      would consume neither. Registration still requires explicit
      ``cell_line_contract`` / ``drug_contract`` (typically
      ``FeatureFormat.NUMERIC_MATRIX``).

      .. code-block:: python

         from drevalpy.components.core.contracts.contracts import FeatureFormat
         from drevalpy.components.core.batch.model_input_batch import ModelInputBatch
         from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
         from drevalpy.registry.predictor import register as register_predictor


         @register_predictor(
             "toyMean",
             description="Predict the training mean response.",
             cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
             drug_contract=FeatureFormat.NUMERIC_MATRIX,
         )
         class ToyMeanPredictor(FeatureFreePredictor):
             def fit(self, batch: ModelInputBatch) -> None:
                 if batch.response is None:
                     raise ValueError("response required")
                 self._mean = float(np.mean(batch.response))

             def predict(self, batch: ModelInputBatch) -> np.ndarray:
                 return np.full(batch.n_pairs, self._mean, dtype=np.float64)

             def get_state(self) -> dict[str, object]:
                 return {"mean": self._mean} if hasattr(self, "_mean") else {}

             def set_state(self, state: dict[str, object]) -> None:
                 if "mean" in state:
                     self._mean = float(state["mean"])

             def is_fitted(self) -> bool:
                 return hasattr(self, "_mean")

   .. tab-item:: Matrix

      ``MatrixPredictor`` flattens the batch with ``batch.to_feature_matrix()``.
      Implement ``_fit_matrix`` / ``_predict_matrix`` on the dense pair-level
      design matrix (the pattern used by ElasticNet, RandomForest, …).

      .. code-block:: python

         from typing import Any

         from sklearn.linear_model import Ridge

         from drevalpy.components.core.contracts.contracts import FeatureFormat
         from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
         from drevalpy.registry.predictor import register as register_predictor


         @register_predictor(
             "toyRidge",
             description="Ridge on concatenated dense cell-line and drug features.",
             cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
             drug_contract=FeatureFormat.NUMERIC_MATRIX,
         )
         class ToyRidgePredictor(MatrixPredictor):
             @classmethod
             def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
                 return {
                     "alpha": {
                         "type": "float",
                         "low": 1e-4,
                         "high": 10.0,
                         "log": True,
                         "default": 1.0,
                     },
                 }

             def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
                 self._estimator = Ridge(alpha=float(self._hyperparameters["alpha"]))
                 self._estimator.fit(x, y)

             def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
                 return np.asarray(self._estimator.predict(x), dtype=np.float64)

             def get_state(self) -> dict[str, object]:
                 return {
                     "estimator": getattr(self, "_estimator", None),
                     "hyperparameters": dict(self._hyperparameters),
                 }

             def set_state(self, state: dict[str, object]) -> None:
                 self._estimator = state["estimator"]
                 self._hyperparameters = dict(state["hyperparameters"])

             def is_fitted(self) -> bool:
                 return getattr(self, "_estimator", None) is not None

   .. tab-item:: Block

      ``BlockPredictor`` reads side-specific
      or named featurizer blocks from ``batch.cell_line_blocks`` /
      ``batch.drug_blocks``. Contracts still constrain the **format** of each
      side; ``required_cell_line_blocks`` / ``required_drug_blocks`` further
      require named views in the stack.

      .. code-block:: python

         from typing import ClassVar

         from sklearn.linear_model import Ridge

         from drevalpy.components.core.contracts.contracts import FeatureFormat
         from drevalpy.components.core.batch.model_input_batch import ModelInputBatch
         from drevalpy.components.predictors.abstract.block import BlockPredictor
         from drevalpy.registry.predictor import register as register_predictor


         @register_predictor(
             "toyBlockRidge",
             description="Ridge on a named expression block plus drug features.",
             cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
             drug_contract=FeatureFormat.NUMERIC_MATRIX,
         )
         class ToyBlockRidgePredictor(BlockPredictor):
             required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("expression",)

             def _fit(self, batch: ModelInputBatch) -> None:
                 x = batch.cell_line_blocks["expression"].values[batch.cell_line_pair_idx]
                 if batch.drug_features is not None and batch.drug_pair_idx is not None:
                     x = np.hstack([x, batch.drug_features[batch.drug_pair_idx]])
                 self._estimator = Ridge(alpha=1.0)
                 self._estimator.fit(x, batch.response)

             def predict(self, batch: ModelInputBatch) -> np.ndarray:
                 x = batch.cell_line_blocks["expression"].values[batch.cell_line_pair_idx]
                 if batch.drug_features is not None and batch.drug_pair_idx is not None:
                     x = np.hstack([x, batch.drug_features[batch.drug_pair_idx]])
                 return np.asarray(self._estimator.predict(x), dtype=np.float64)

             def get_state(self) -> dict[str, object]:
                 return {"estimator": getattr(self, "_estimator", None)}

             def set_state(self, state: dict[str, object]) -> None:
                 self._estimator = state["estimator"]

             def is_fitted(self) -> bool:
                 return getattr(self, "_estimator", None) is not None

Custom splitters
----------------

Register a splitter function under a mode name with
``@drevalpy.registry.splitter.register``. The function must accept the
splitter protocol signature and return a list of
:class:`~drevalpy.types.SplitMasks`:

.. code-block:: python

   from drevalpy.registry.splitter import register as register_splitter
   from drevalpy.types import MuDataLike, SplitMasks


   @register_splitter("MY_LCO", "Custom LCO with 80/20 fraction", validation="LCO")
   def my_lco(
       mudataset: MuDataLike,
       n_splits: int = 5,
       validation_ratio: float = 0.1,
       random_state: int = 42,
   ) -> list[SplitMasks]:
       # ... custom splitting logic ...
       return folds

The ``validation`` parameter specifies which leakage constraint to enforce
automatically after every split (``"LCO"``, ``"LDO"``, ``"LPO"``, or
``"LTO"``). If validation fails, a ``SplitValidationError`` is raised.

Once registered, your mode can be used anywhere a mode string is accepted:

.. code-block:: python

   from drevalpy.data import split

   folds = split(dataset, mode="MY_LCO", n_splits=5)

Custom visualizations
---------------------

Register a visualization class with
``@drevalpy.registry.visualization.register``. The class must implement the
``Visualization`` base interface (``compute`` and ``to_multiqc`` methods):

.. code-block:: python

   from drevalpy.registry.visualization import register as register_visualization
   from drevalpy.visualization.base import Visualization


   @register_visualization(
       "my_scatter",
       "Custom scatter plot of predictions vs response.",
       result_type="ExperimentResult",
       requirements=frozenset(),
   )
   class MyScatterPlot(Visualization):
       def compute(self, result, *, dataset=None):
           # Extract data from ExperimentResult or ModelResult
           ...

       def to_multiqc(self):
           # Return list of MultiQC section objects
           ...

The ``result_type`` declares whether this visualization operates on an
``ExperimentResult`` (aggregated across models) or a ``ModelResult``
(single model). The ``requirements`` frozenset specifies conditions that must
be met for the report system to select this visualization automatically (for
example: multiple CV folds, multiple models, or a reference model).

Custom dataset sources
----------------------

Register remote or local storage locations as **sources**, then point named
datasets at files under those sources:

.. code-block:: python

   from drevalpy.registry.dataset import register_source, register_dataset

   register_source(
       "my_s3_bucket",
       "s3://my-bucket/datasets/",
       storage_options={"key": "...", "secret": "..."},
   )

   register_dataset("MyScreen", source="my_s3_bucket", file="MyScreen.h5mu")

The two-level design means you register a source once and then add as many
dataset entries under it as needed. Any protocol that
`fsspec <https://filesystem-spec.readthedocs.io/>`_ supports works: HTTPS,
S3, GCS, Azure Blob Storage, or local file paths. Once registered, load by
name as usual:

.. code-block:: python

   from drevalpy.data import load

   dataset = load("MyScreen")

Literature references
---------------------

``LiteratureReference`` is optional **provenance metadata** for components
ported from a paper or external repository. Pass it as ``reference=...`` on
the register decorator. It does **not** change training, composition checks,
or checkpoints — it only documents where the idea came from.

.. code-block:: python

   from drevalpy.types import LiteratureReference

   TOY_RIDGE_REFERENCE = LiteratureReference(
       repo_url="https://github.com/example/toy-ridge",
       citation_doi="10.1234/example",
       citation_text="Example ridge baseline for documentation.",
       deviations=(
           "Uses sklearn Ridge on flattened features; "
           "hyperparameter defaults differ from the upstream script."
       ),
   )


   @register_predictor(
       "toyRidge",
       description="Ridge on concatenated dense features.",
       reference=TOY_RIDGE_REFERENCE,
   )
   class ToyRidgePredictor(MatrixPredictor):
       ...

Loading extensions
------------------

Import your components
~~~~~~~~~~~~~~~~~~~~~~

``@register_*`` runs when the module is imported. If your package is
installable (or otherwise on ``PYTHONPATH``), a normal import is enough:

.. code-block:: python

   import my_components.toy_featurizer  # registers toyCellLine
   import my_components.toy_predictors  # registers toyMean, toyRidge, …

   from drevalpy.models import construct_model

   ToyRidge = construct_model(
       "ToyRidge",
       "toyCellLine:fingerprints:toyRidge",
   )
   model = ToyRidge()

Other sources: ``load_extensions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`~drevalpy.registry.load_extensions` when components are not a
normal importable package, or when you also want to register external zoo
YAML in one call:

- ``modules`` — dotted names (same effect as ``import``)
- ``files`` — individual ``.py`` paths executed as temporary modules
- ``directories`` — all ``*.py`` in a folder (non-recursive; ``__init__.py``
  skipped; sorted by filename)
- ``zoo_files`` — YAML presets that map a **name** to an already-registered
  stack (not Python classes, not experiment hpam YAML)

.. code-block:: python

   from drevalpy.registry import load_extensions
   from drevalpy.models import construct_model

   load_extensions(
       directories=["my_components"],
       zoo_files=["my_zoo/toy.yaml"],
   )
   ToyMean = construct_model("toyMean")  # zoo preset name

Plugin discovery
~~~~~~~~~~~~~~~~

When the package is imported, it scans for installed Python packages that
advertise the ``drevalpy.plugins`` entry point group. Importing the advertised
module triggers registration decorators, making a plugin's components
available without any explicit user action beyond installation.

In your plugin's ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."drevalpy.plugins"]
   my_plugin = "my_plugin.components"

Extension directories
~~~~~~~~~~~~~~~~~~~~~

Both the CLI and the Python API accept an **extensions directory** containing
``.py`` and ``.yaml`` files. All Python files in the directory are imported
(triggering registration decorators for any registry), and all YAML files are
loaded as model-zoo presets or dataset declarations. An environment variable
(``DREVALPY_EXTENSIONS_DIR``) provides the same mechanism without requiring a
CLI flag.

Saving and loading with custom components
-----------------------------------------

Checkpoints are ZIP archives that store the resolved ``ModelConfig`` (component
**names**) and fitted state — not the Python classes themselves. On load,
DrEvalPy looks those names up in the registries again, then restores state. If
a custom featurizer or predictor is not registered in the process that calls
``load`` / ``load_model``, reconstruction fails.

Import the same modules (or call ``load_extensions``) before loading:

.. code-block:: python

   import my_components.toy_featurizer
   import my_components.toy_predictors

   from drevalpy.models import load_model

   model = load_model("checkpoints/toy_ridge.zip")

Built-in zoo models need no extra step; only custom component names require
this. See :doc:`models` for the general save/load lifecycle.
