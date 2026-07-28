Model architecture overview
===========================

For the interface-neutral overview of recipes, concatenation, and
hyperparameter spaces, see :doc:`/concepts/from_components_to_models`. This page covers the
Python orchestration layer in more detail.

DrEvalPy has two cooperating layers:

1. **Component stack** under ``drevalpy.components`` with featurizers,
   predictors, registries, and tuning helpers.
2. **Public orchestration** under ``drevalpy.models`` with declarative
   ``ModelConfig``, zoo YAML, and ``construct_model`` returning thin
   ``DRPModel`` subclasses.

Typical composition:

.. code-block:: text

   ModelConfig -> construct_model(name[, spec]) -> DRPModel subclass -> instance
   (instance materializes featurizer(s) + predictor as an internal component stack)

Predictor input batch
---------------------

Training and prediction always build a single ``ModelInputBatch`` before calling
``predictor.fit(batch)`` or ``predictor.predict(batch)``. The batch carries
pair identifiers, optional response values, entity-level feature matrices,
named featurizer blocks, optional raw ``FeatureDataset`` inputs,
early-stopping response data, and a small ``TrainingContext`` (checkpoint
directory / logging metadata).

- Matrix predictors flatten the batch with ``batch.to_feature_matrix()``.
- Block predictors read ``batch.cell_line_blocks`` and ``batch.drug_blocks``.
- Baseline predictors use pair identifiers and/or response values only.
- Literature predictors that declare ``requires_raw_feature_datasets`` read
  the raw ``FeatureDataset`` inputs carried on the batch.

Resolving built-in models
-------------------------

Prefer zoo names with ``construct_model``:

.. code-block:: python

   from drevalpy.models import construct_model
   from drevalpy.models.zoo import list_zoo_names
   from drevalpy.types.model_scope import ModelScope

   ElasticNet = construct_model("ElasticNet")
   model = ElasticNet()
   model.train(...)
   model.predict(...)

   # Discover presets by scope
   single_drug = list_zoo_names(scope=ModelScope.SINGLE_DRUG)

When you already hold a ``ModelConfig``, pass it as the second argument:

.. code-block:: python

   from drevalpy.models import construct_model
   from drevalpy.models.config import ModelConfig

   config = ModelConfig.from_spec("ElasticNet")
   ElasticNet = construct_model("ElasticNet", config)
   model = ElasticNet()

Programmatic composition
------------------------

For custom stacks without adding a zoo file, pass a recipe as the second
argument:

.. code-block:: python

   from drevalpy.models import construct_model
   from drevalpy.models.config import ModelConfig

   CustomElasticNet = construct_model(
       "myElasticNet",
       "scaledGeneExpression:fingerprints:elasticNet",
   )
   model = CustomElasticNet({"alpha": 0.1, "l1_ratio": 0.5})

   # Same stack, config object instead of recipe string
   config = ModelConfig.from_spec("scaledGeneExpression:fingerprints:elasticNet")
   CustomElasticNet2 = construct_model("myElasticNet", config)

``construct_model(name)`` / ``construct_model(name, spec)`` return a **class**.
Call it with optional flat hyperparameters to get a **fresh instance**.

Explicit omics view grammar
---------------------------

Cell-line featurizers that operate on a single omics layer use bracket syntax:

.. code-block:: text

   raw[expression]+pca[proteomics]:identity:randomForest

- ``raw[view]`` passes through one dense omics view without preprocessing.
- ``pca[view]`` applies PCA to one dense omics view. The view is required.
- ``+`` concatenates featurizers into ``concatFeaturizers``.

Supported view aliases include ``expression`` (gene expression),
``methylation``, ``mutations``, ``proteomics``, and ``cnv`` (copy-number
variation). YAML presets use the same atoms.

Featurizer hyperparameter tuning (dotted keys)
----------------------------------------------

Ray/Optuna search spaces use **dotted keys** that mirror the composed stack.
Predictor parameters look like ``predictor.<registryName>.<param>``; featurizer
parameters look like
``featurizer.<registry>.<qualifiedFeaturizer>.<param>``, where the qualified
name matches the recipe atom (for example ``pca[expression]`` or
``landmarkGenes``).

Examples:

.. code-block:: text

   predictor.elasticNet.alpha
   predictor.elasticNet.l1_ratio
   featurizer.cell_line.pca[expression].n_components
   featurizer.cell_line.landmarkGenes.standardize

The same base featurizer on different views stays independently tunable
(``pca[expression]`` vs ``pca[proteomics]``). Repeating the same qualified
selector in one slot is rejected.

Flat constructor dicts remain supported for predictor keys such as
``alpha`` and legacy featurizer aliases (``methylation_n_components``).
Structured overrides may also use dotted keys directly.

Feature contracts and validation limits
---------------------------------------

Each featurizer declares a ``FeatureKind`` (``dense``, ``graph``, or
``sequence``) via its registration decorator. Predictors declare
``cell_line_contract`` and ``drug_contract``. ``ModelConfig`` validation
checks that featurizer output kinds match predictor input kinds.

``FeatureContract`` currently compares **only the broad ``FeatureKind``**.
Graph compatibility is therefore ``graph`` expected and ``graph`` provided —
finer details such as node feature dimension or edge semantics are **not**
validated yet. Extend contracts carefully when pairing new featurizers with
structured predictors.

Scope and early stopping
------------------------

- Multi-drug is the default model scope and requires a drug featurizer for
  feature-based predictors.
- Single-drug models set ``scope: single_drug`` in their zoo YAML; the class
  exposes ``is_single_drug_model`` for experiment routing.
- Early stopping is derived from predictor capability metadata
  (``supports_early_stopping``) via the zoo predictor name.

Predictors and construction
---------------------------

Featurizers receive constructor kwargs from ``FeaturizerConfig``. Predictors
receive static hyperparameters from ``PredictorConfig.create_instance()``.
``DRPModel.train`` fits featurizers, builds a ``ModelInputBatch``, and
calls ``predictor.fit`` — there is no public ``Predictor.build``. Dimension
allocation that depends on fitted features happens inside ``fit``.

Persistence
-----------

Native checkpoints store a versioned payload with the resolved ``ModelConfig``
and fitted component state (``model.joblib``, format ``drevalpy-model``). Run
metadata and CV splits live beside checkpoints, not inside them. Legacy
checkpoint formats and deep model import paths are unsupported; see
:doc:`persistence` and :doc:`custom_models`.

Extension path
--------------

Register featurizers/predictors, compose a ``ModelConfig`` or zoo YAML, and
use ``construct_model`` / ``load_extensions``. Direct ``DRPModel`` subclass
authoring is not the supported extension mechanism. See :doc:`custom_models`
for a full example, and :doc:`component_catalog` for built-in registry names.

Backward compatibility
----------------------

Factory dictionaries and view keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before 1.6.0, factory dictionaries and flat view keys were the usual
interfaces. Factory dicts remain as lazy built-in-only compatibility views
(equal to ``construct_model(name)`` for zoo names) but emit ``FutureWarning``
and may be removed in a future release:

- ``MODEL_FACTORY``, ``MULTI_DRUG_MODEL_FACTORY``, ``SINGLE_DRUG_MODEL_FACTORY``
  — prefer ``construct_model``, ``ModelConfig.from_spec``, and
  ``list_zoo_names(scope=...)`` instead.
- Flat ``cell_line_views`` / ``drug_views`` in constructor / hpam YAML —
  set ``cell_line_featurizer`` / ``drug_featurizer`` in zoo YAML or a
  recipe string instead (see :doc:`model_inputs`).

Named root exports and ``ModelConfig.create_model()`` are removed; see
:doc:`models`.

No longer supported
~~~~~~~~~~~~~~~~~~~

Deep imports such as ``drevalpy.models.DIPK.dipk`` or
``drevalpy.models.baselines.*`` no longer resolve. Legacy checkpoint formats
(including ``composed_model.joblib``) are not loadable.
