Run your own model
===================

DrEvalPy models are composed from registered **featurizers** and **predictors**.
Do not subclass ``DRPModel`` directly for new models. Register components, describe
the stack with a ``ModelConfig`` or zoo preset, and expose a generated facade through
``construct_model``.

High-level path
---------------

1. Register a featurizer and/or predictor under ``drevalpy.components``.
2. Describe the stack with a recipe string, YAML zoo entry, or ``ModelConfig`` dict.
3. Resolve it with ``construct_model`` or ``ModelConfig.from_spec``.

Minimal complete extension example
----------------------------------

The following end-to-end sketch registers an external cell-line featurizer and
predictor, loads a zoo preset, builds a public ``DRPModel`` facade, and wires
Ray/Optuna tuning through the normal experiment API.

**1. Component module** (``my_components/toy_stack.py``):

.. code-block:: python

    from __future__ import annotations

    import numpy as np

    from drevalpy.components.contracts import FeatureKind
    from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
    from drevalpy.components.model_input_batch import ModelInputBatch
    from drevalpy.components.predictors.baseline import BaselinePredictor
    from drevalpy.components.registry import register_cell_line_featurizer, register_predictor


    @register_cell_line_featurizer(
        "toyCellLine",
        description="Constant cell-line features for demos.",
        category="general_purpose",
        contract=FeatureKind.DENSE,
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


    @register_predictor(
        "toyPredictor",
        description="Predict the training mean response.",
        category="general_purpose",
    )
    class ToyPredictor(BaselinePredictor):
        def fit(self, batch: ModelInputBatch) -> None:
            if batch.response is None:
                msg = "response required"
                raise ValueError(msg)
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

Registration decorators attach metadata (name, description, category, and optional
``FeatureKind`` contract) to the class. Fitted components must implement
``get_state`` / ``set_state`` so ``ComposedModel`` checkpoints round-trip.

**2. External zoo YAML** (``my_zoo/toy.yaml``):

.. code-block:: yaml

    toyMean:
      cell_line_featurizer: toyCellLine
      predictor: toyPredictor

**3. Load extensions and construct the model**:

.. code-block:: python

    from drevalpy.components import load_extensions
    from drevalpy.models import construct_model
    from drevalpy.models.config import ModelConfig

    load_extensions(
        directories=["my_components"],
        zoo_files=["my_zoo/toy.yaml"],
    )

    # Recipe string: cellLineFeaturizer:drugFeaturizer:predictor
    ToyMean = construct_model("ToyMean", "toyCellLine:identity:toyPredictor")

    # Or resolve the zoo entry by name
    ToyMeanZoo = construct_model("toyMean")
    config = ModelConfig.from_spec("toyMean")
    composed = config.create_model()

``construct_model`` yields a ``DRPModel`` facade with the usual ``build_model``,
``train``, ``predict``, ``save``, and ``load`` lifecycle.
``ModelConfig.create_model()`` returns the underlying ``ComposedModel``.

**4. Tuning** (structured dotted keys, Ray + Optuna):

.. code-block:: python

    from drevalpy.experiment import drug_response_experiment

    drug_response_experiment(
        models=[ToyMean],
        response_data=...,
        hyperparameter_tuning=True,
        hpo_num_samples=16,
        hpo_random_state=42,
    )

When ``hyperparameter_tuning=False``, experiments use each predictor's default
hyperparameters only (``get_default_hyperparameters()``). See
:doc:`hyperparameter_migration` for migrating off old YAML grids.

Zoo presets
-----------

Built-in models live under ``drevalpy/models/zoo/*.yaml``. Each file is one
factory name. Single-drug models set ``scope: single_drug``. Early stopping is
derived from predictor capability metadata.

Example zoo entry:

.. code-block:: yaml

    cell_line_featurizer: scaledGeneExpression
    drug_featurizer: fingerprints
    predictor: elasticNet

Flat hyperparameters
--------------------

Public ``build_model`` accepts a flat hyperparameter dict. Overrides are
translated onto the resolved ``ModelConfig`` (predictor and featurizer local
keys). Flat ``cell_line_views`` / ``drug_views`` still work but are
**deprecated** — prefer zoo featurizer blocks or recipe strings
(:doc:`example_flexible_inputs`).

Deprecated compatibility catalogs
---------------------------------

``MODEL_FACTORY``, ``MULTI_DRUG_MODEL_FACTORY``, and
``SINGLE_DRUG_MODEL_FACTORY`` remain importable for compatibility but emit a
``FutureWarning``. Prefer:

.. code-block:: python

    from drevalpy.models import construct_model, ElasticNetModel
    from drevalpy.models.config import ModelConfig
    from drevalpy.models.zoo import list_zoo_names
    from drevalpy.types.model_scope import ModelScope

    ElasticNet = construct_model("ElasticNet")  # or ElasticNetModel
    composed = ModelConfig.from_spec("ElasticNet").create_model()
    single_drug = list_zoo_names(scope=ModelScope.SINGLE_DRUG)

Named root exports, ``construct_model``, experiment routing, and CLI model
names continue to work.

Breaking changes (modularity release)
-------------------------------------

The following **are removed** and require migration:

Deep imports
~~~~~~~~~~~~

Paths such as ``drevalpy.models.DIPK.dipk`` or ``drevalpy.models.baselines.*``
no longer resolve. Import catalog models from the package root instead:

.. code-block:: python

    from drevalpy.models import DIPKModel, ElasticNetModel, construct_model

    model = construct_model("ElasticNet")()

For component-level work (custom featurizers/predictors), use ``drevalpy.components``
and the registry helpers documented in :doc:`model_architecture`.

Legacy checkpoints
~~~~~~~~~~~~~~~~~~

Checkpoints saved before the component-native stack (pickled ``.model`` attributes,
standalone scalers, naive mean buffers, etc.) are **not** loadable. Retrain models
with the current release and persist via the native payload:

- ``composed_model.joblib`` — resolved ``ModelConfig`` plus fitted component state
- ``hyperparameters.json`` — public flat hyperparameters used for ``build_model``

Hyperparameter tuning behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``get_hyperparameter_set()`` returns **one** default dict per model, not a full
  YAML ``ParameterGrid``. Use ``hyperparameter_tuning=True`` with ``hpo_num_samples``
  for search, or ``get_structured_hyperparameter_space()`` for programmatic access.
- ``hyperparameter_tuning=False`` uses predictor defaults only; it does **not** mean
  "debug mode" and does **not** silently iterate an old grid.
- ``multiprocessing=True`` is a deprecated alias for the Ray Tune path; prefer
  ``hyperparameter_tuning=True`` and ``hpo_num_samples``.

Dependencies
~~~~~~~~~~~~

``pydantic``, ``optuna``, and ``ray[tune]`` are **core** dependencies (not optional
extras). ``ModelConfig`` validation uses Pydantic v2 with ``extra="forbid"`` on
config models — unknown YAML keys raise validation errors rather than being ignored.

See also :doc:`hyperparameter_migration` and :doc:`installation`.

Unsupported extension paths
---------------------------

The following are intentionally **not** supported:

* Documented ``DRPModel`` subclass authoring as the extension path
* Fitted-state introspection on legacy attributes (``.model``, private scalers, naive means)
* Loading checkpoints produced before the ``ComposedModel`` persistence format
