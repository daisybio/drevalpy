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
named featurizer blocks, early-stopping response data, and a small
``TrainingContext`` (checkpoint directory / logging metadata).

- ``MatrixPredictor`` flattens the batch with ``batch.to_feature_matrix()``.
- ``BlockPredictor`` (alias ``StructuredPredictor``) reads side-specific or
  named featurizer blocks.
- ``FeatureFreePredictor`` uses pair identifiers and/or response values only.

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
``cell_line_featurizer.<qualifiedFeaturizer>.<param>`` or
``drug_featurizer.<qualifiedFeaturizer>.<param>``, where the qualified
name matches the recipe atom (for example ``pca[expression]`` or
``landmarkGenes``).

Examples:

.. code-block:: text

   predictor.elasticNet.alpha
   predictor.elasticNet.l1_ratio
   cell_line_featurizer.pca[expression].n_components
   cell_line_featurizer.landmarkGenes.standardize

The same base featurizer on different views stays independently tunable
(``pca[expression]`` vs ``pca[proteomics]``). Repeating the same qualified
selector in one slot is rejected.

Flat constructor dicts remain supported for predictor keys such as
``alpha`` and legacy featurizer aliases (``methylation_n_components``).
Structured overrides may also use dotted keys directly.

Feature formats, interfaces, and validation
-------------------------------------------

Each featurizer declares a ``FeatureFormat`` (``numeric_matrix``, ``graph``,
or ``ragged_sequence``). Predictors declare ``cell_line_contract`` and
``drug_contract`` plus exactly one input interface
(``FeatureFreePredictor``, ``MatrixPredictor``, or ``BlockPredictor``).
``ModelConfig`` validation checks formats and interface rules; discovery tags
and literature references are descriptive only.

Graph and ragged payloads are not numeric matrices. Matrix predictors reject
them. Block predictors can consume those payloads and validate their structure;
for example, DrugGNN validates PyG ``Data`` objects (``x``, ``edge_index``,
consistent node-feature dimensions).

Scope and early stopping
------------------------

- Multi-drug is the default model scope and requires a drug featurizer for
  matrix/block predictors that consume the drug side.
- Feature-based single-drug models use the ``identity`` drug featurizer to
  create and route per-drug estimators without adding identity columns to
  their design matrices. A three-slot recipe infers ``scope: single_drug``
  from the predictor's sole supported scope.
- Literature predictors declare the fitted featurizer blocks they need; the
  single-drug MOLIR and SuperFELTR presets use ``identity`` for drug routing.
- Feature-free stacks skip featurizer fit/transform entirely.
- Early stopping is derived from predictor capability metadata
  (``supports_early_stopping``) via the zoo predictor name.
- Default prediction mode is regression; classification requires an explicit
  predictor opt-in.

Predictors and construction
---------------------------

Featurizers receive constructor kwargs from ``CellLineFeaturizerConfig`` /
``DrugFeaturizerConfig`` (subclasses of ``FeaturizerConfig``). Predictors
receive static hyperparameters from ``PredictorConfig.create_instance()``.
``DRPModel.train`` fits featurizers, builds a ``ModelInputBatch``, and
calls ``predictor.fit`` — there is no public ``Predictor.build``. Dimension
allocation that depends on fitted features happens inside ``fit``.

Literature predictor ownership
------------------------------

Each literature model lives in its own package below
``drevalpy.components.predictors.literature``. Its ``predictor.py`` directly
implements one input interface and owns feature loading, validation, fitting,
prediction, and persistence. Model-specific networks and data/training helpers
are sibling modules in the same package. The literature root contains only
behavior-neutral helpers such as raw-view validation, block conversion, and
reference metadata; there is no shared lifecycle engine or string-based engine
resolver.

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
for a full example, and :doc:`/concepts/component_catalog` for built-in registry names.

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
