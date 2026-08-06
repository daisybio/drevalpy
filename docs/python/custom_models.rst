Custom Components and Models
============================

If you are reading this, we assume you are already familiar with:

- :doc:`models` — ``construct_model``, recipes, ``ModelConfig``, and lifecycle
- :doc:`/concepts/component_catalog`
- :doc:`/concepts/from_components_to_models`

Built-in models are stacks of registered **featurizers** and **predictors**.
To add something new, register custom components first, then compose them
the same way as any zoo preset or recipe. Do not subclass ``DRPModel``
directly.

Registering custom components
-----------------------------

Components live under ``drevalpy.components``. Decorators register a class by
name and attach metadata (description, optional ``tags``, optional
``LiteratureReference``, and role-specific ``FeatureFormat`` contracts).
Fitted components must implement ``get_state`` / ``set_state`` so ``*.zip``
checkpoints round-trip.

Custom featurizers
~~~~~~~~~~~~~~~~~~

Subclass ``CellLineFeaturizer`` or ``DrugFeaturizer`` and register with
``@register_cell_line_featurizer`` or ``@register_drug_featurizer``.

Declare a ``FeatureFormat`` **contract** on registration
(``numeric_matrix``, ``graph``, or ``ragged_sequence``). That is the payload
format this featurizer produces. Composition validation compares it to the
predictor's ``cell_line_contract`` / ``drug_contract`` and rejects stacks
where the formats disagree (for example a ``graph`` drug featurizer with a
predictor that expects ``numeric_matrix`` on the drug side). Registry names
and the format vocabulary are listed in
:doc:`/concepts/component_catalog`.

.. code-block:: python

   from __future__ import annotations

   import numpy as np

   from drevalpy.components.contracts import FeatureFormat
   from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
   from drevalpy.components.registry import register_cell_line_featurizer


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

Custom predictors
~~~~~~~~~~~~~~~~~

Every predictor must inherit exactly one input interface and register with
``@register_predictor``. The available types were already introduced in
:doc:`/concepts/component_catalog`.
The details about what the input for each predictor type looks like are explained in the tab switcher below.

.. tab-set::

   .. tab-item:: Feature-free

      ``FeatureFreePredictor`` uses pair identifiers and/or response values
      only. Composition forbids cell-line and drug featurizers for it, since it
      would consume neither. Registration still requires explicit
      ``cell_line_contract`` / ``drug_contract`` (typically
      ``FeatureFormat.NUMERIC_MATRIX``).

      .. code-block:: python

         from drevalpy.components.contracts import FeatureFormat
         from drevalpy.components.model_input_batch import ModelInputBatch
         from drevalpy.components.predictors.feature_free import FeatureFreePredictor
         from drevalpy.components.registry import register_predictor


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
      Dense tabular models declare ``numeric_matrix`` on both sides, so they
      pair with featurizers such as ``toyCellLine`` and ``fingerprints``.

      .. code-block:: python

         from typing import Any

         from sklearn.linear_model import Ridge

         from drevalpy.components.contracts import FeatureFormat
         from drevalpy.components.predictors.matrix import MatrixPredictor
         from drevalpy.components.registry import register_predictor


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
      require named views in the stack (for example an ``expression`` block
      from ``raw[expression]``).

      .. code-block:: python

         from typing import ClassVar

         from sklearn.linear_model import Ridge

         from drevalpy.components.contracts import FeatureFormat
         from drevalpy.components.model_input_batch import ModelInputBatch
         from drevalpy.components.predictors.block import BlockPredictor
         from drevalpy.components.registry import register_predictor


         @register_predictor(
             "toyBlockRidge",
             description="Ridge on a named expression block plus drug features.",
             cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
             drug_contract=FeatureFormat.NUMERIC_MATRIX,
         )
         class ToyBlockRidgePredictor(BlockPredictor):
             required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("expression",)

             def fit(self, batch: ModelInputBatch) -> None:
                 if batch.response is None:
                     raise ValueError("response required")
                 x = batch.cell_line_blocks["expression"].values[batch.cell_line_pair_idx]
                 if batch.drug_features is not None and batch.drug_pair_idx is not None:
                     x = np.hstack([x, batch.drug_features[batch.drug_pair_idx]])
                 y = np.asarray(batch.response, dtype=np.float64)
                 self._estimator = Ridge(alpha=1.0)
                 self._estimator.fit(x, y)

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

Feature-free predictors need only a predictor token in ``construct_model``.
Matrix and block predictors pair with featurizers whose ``contract`` matches
the predictor's ``cell_line_contract`` / ``drug_contract`` (see below).

Deprecated: FeatureDataset predictor bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do **not** subclass ``FeatureDatasetBlockPredictor`` or
``SingleDrugBlockPredictor`` for new components. Those bases are a
**deprecated** adapter for literature (and similar) cores that still call
``train`` / ``predict`` with ``FeatureDataset`` after the stack has already
built a ``ModelInputBatch``. Prefer the three interfaces above:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Need
     - Use
     - Input
   * - Response / ids only
     - ``FeatureFreePredictor``
     - ``ModelInputBatch``
   * - Flattened dense pair features
     - ``MatrixPredictor`` (or sklearn tabular bases)
     - ``ModelInputBatch``
   * - Named / typed blocks (graphs, multi-view, …)
     - ``BlockPredictor`` directly
     - ``batch.cell_line_blocks`` / ``drug_blocks``
   * - Per-drug dense estimators
     - ``SingleDrugSklearnPredictor`` pattern
     - ``ModelInputBatch`` (not the FeatureDataset bridge)

``FeatureDataset`` remains the correct type for loading raw entity views into
``DRPModel`` / featurizers. Only rebuilding ``FeatureDataset`` *inside* a
predictor is deprecated.

Literature references
~~~~~~~~~~~~~~~~~~~~~

``LiteratureReference`` is optional **provenance metadata** for components
ported from a paper or external repository. Pass it as ``reference=...`` on
the register decorator. It does **not** change training, composition checks,
or checkpoints — it only documents where the idea came from and how the
DrEvalPy port differs from the upstream code. When you set a reference, all
of these fields are required and validated:

- ``repo_url`` — upstream implementation (``http://`` or ``https://``)
- ``citation_text`` and/or ``citation_doi`` — how to cite the method
- ``deviations`` — intentional differences from the reference (preprocessing,
  packaging, defaults, missing pieces, …)

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

Built-in literature models (DIPK, SparseGO, …) attach references the same way;
see their entries in the component catalog.

Discovery
~~~~~~~~~

Inspect what is registered with the role-specific listing helpers (also
exported from ``drevalpy.components``):

- :func:`~drevalpy.components.list_cell_line_featurizer_metadata`
- :func:`~drevalpy.components.list_drug_featurizer_metadata`
- :func:`~drevalpy.components.list_predictor_metadata`

Each returns a list of dicts with name, description, tags, literature reference
fields (for example ``repo_url``, ``citation``, ``citation_doi``), and either
``output_format`` (featurizers) or ``input_interface`` (predictors). Pass
``tag=...`` to keep only matching
entries (for example ``tag="baseline"``). Tags are discovery filters only and
never change validation. The generated
:doc:`/concepts/component_catalog` is built from the same metadata.

.. code-block:: python

   from drevalpy.components import (
       list_cell_line_featurizer_metadata,
       list_predictor_metadata,
   )

   predictors = list_predictor_metadata()
   baselines = list_predictor_metadata(tag="baseline")
   cell_line = list_cell_line_featurizer_metadata()

Composing models from custom components
---------------------------------------

Once components are registered, compose them exactly as on :doc:`models`: a
recipe string, zoo YAML, or ``ModelConfig``, then ``construct_model``.

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

Feature-free predictors need only a predictor token
(``construct_model("ToyMean", "toyMean")``). Block predictors follow the same
recipe form with the views they require (for example
``raw[expression]:fingerprints:toyBlockRidge``).

You can also put named presets in a zoo YAML file and load them (see below),
but recipes and ``ModelConfig`` work immediately after import — no extra
loader step.

Other sources: ``load_extensions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`~drevalpy.components.load_extensions` when components are not a
normal importable package, or when you also want to register external zoo
YAML in one call:

- ``modules`` — dotted names (same effect as ``import``)
- ``files`` — individual ``.py`` paths executed as temporary modules
- ``directories`` — all ``*.py`` in a folder (non-recursive; ``__init__.py``
  skipped; sorted by filename)
- ``zoo_files`` — YAML presets that map a **name** to an already-registered
  stack (not Python classes, not experiment hpam YAML)

.. code-block:: text

   my_components/          # -> directories=[...]  (or import as a package)
     toy_featurizer.py
     toy_predictors.py
   my_zoo/
     toy.yaml              # -> zoo_files=[...]

.. code-block:: python

   from drevalpy.components import load_extensions
   from drevalpy.models import construct_model

   load_extensions(
       directories=["my_components"],
       zoo_files=["my_zoo/toy.yaml"],
   )
   ToyMean = construct_model("toyMean")  # zoo preset name

Example zoo YAML:

.. code-block:: yaml

   toyMean:
     predictor: toyMean

   toyRidge:
     cell_line_featurizer: toyCellLine
     drug_featurizer: fingerprints
     predictor: toyRidge

Run the resulting class through :doc:`experiments` the same way as any zoo
preset.

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
this. See :doc:`models` for the general save/load lifecycle (``.zip`` is
appended automatically when the path does not already end with it).
