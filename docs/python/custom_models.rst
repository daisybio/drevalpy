Custom models
=============

If you are reading this, we assume you are already familiar with these
concepts:

- :doc:`/concepts/component_catalog`
- :doc:`/concepts/from_components_to_models`

DrEvalPy models are composed from registered **featurizers** and
**predictors**. Do not subclass ``DRPModel`` directly for new models. Register
components, describe the stack with a ``ModelConfig`` or zoo preset, and
resolve a public class with :func:`~drevalpy.models.construct_model`.

High-level path
---------------

1. Register a featurizer and/or predictor under ``drevalpy.components``.
2. Describe the stack with a recipe string, YAML zoo entry, or ``ModelConfig``
   dict.
3. Resolve a ``DRPModel`` subclass with ``construct_model(name[, spec])`` and
   construct a fresh instance.

Minimal complete extension example
----------------------------------

The following end-to-end sketch registers an external cell-line featurizer and
predictor, loads a zoo preset, builds a public ``DRPModel`` class, and wires
hyperparameter tuning through the normal experiment API.

**1. Component module** (``my_components/toy_stack.py``):

.. code-block:: python

   from __future__ import annotations

   import numpy as np

   from drevalpy.components.contracts import FeatureFormat
   from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
   from drevalpy.components.model_input_batch import ModelInputBatch
   from drevalpy.components.predictors.feature_free import FeatureFreePredictor
   from drevalpy.components.registry import register_cell_line_featurizer, register_predictor


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


   @register_predictor(
       "toyPredictor",
       description="Predict the training mean response.",
   )
   class ToyPredictor(FeatureFreePredictor):
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

Registration decorators attach metadata (name, description, optional
``tags``, optional ``LiteratureReference``, and role-specific
``FeatureFormat`` contracts) to the class. Fitted components must implement
``get_state`` / ``set_state`` so ``model.joblib`` checkpoints round-trip.

Every predictor must inherit exactly one input interface:

- ``FeatureFreePredictor`` — response/identifiers only; no featurizers
- ``MatrixPredictor`` — one numeric pair-level design matrix
- ``BlockPredictor`` — side-specific or named featurizer blocks

Neural encoders remain private implementation details inside predictors.
For larger predictors, use a predictor-owned package: keep the registered
class and lifecycle orchestration in ``predictor.py`` and place model-specific
networks, datasets, and training helpers in small sibling modules. Shared
predictor-root helpers should be behavior-neutral; lifecycle adapters and
string-based implementation resolvers obscure ownership and are unsupported.

**2. External zoo YAML** (``my_zoo/toy.yaml``):

.. code-block:: yaml

   toyMean:
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

   # Feature-free predictors: predictor-only specs
   ToyMean = construct_model("ToyMean", "toyPredictor")

   # Feature-based models still use the three-slot recipe
   # ToyRF = construct_model("ToyRF", "toyCellLine:identity:randomForest")

   model = ToyMean()

Discovery and literature references
-----------------------------------

Use ``list_*_metadata()`` to inspect registered components. Optional
``tags`` (for example ``baseline``) are discovery filters only and never
change validation. Literature ports attach a ``LiteratureReference`` with
repository URL, citation, and deviations:

.. code-block:: python

   from drevalpy.types import LiteratureReference

   reference = LiteratureReference(
       repo_url="https://github.com/example/repo",
       citation_doi="10.1234/example",
       deviations="Modular port; encoders remain inside the predictor.",
   )
