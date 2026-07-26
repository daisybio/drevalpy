Model architecture overview
===========================

DrEvalPy has two cooperating layers:

1. **Component stack** under ``drevalpy.components`` with featurizers, predictors,
   registries, and tuning helpers.
2. **Public orchestration** under ``drevalpy.models`` with ``ModelConfig``, zoo YAML,
   ``ComposedModel``, and generated ``NativeDRPModel`` facades via ``construct_model``.

Typical composition:

.. code-block:: text

    ModelConfig -> featurizer(s) + predictor -> ComposedModel
    construct_model(name) / named facade -> NativeDRPModel -> same ComposedModel stack

Predictor input batch
---------------------

``ComposedModel`` always builds a single ``ModelInputBatch`` before calling
``predictor.fit(batch)`` or ``predictor.predict(batch)``. The batch carries pair
identifiers, optional response values, entity-level feature matrices, named
featurizer blocks, optional raw ``FeatureDataset`` inputs, early-stopping
response data, and a small ``TrainingContext`` (checkpoint directory / logging
metadata).

- Matrix predictors flatten the batch with ``batch.to_feature_matrix()``.
- Block predictors read ``batch.cell_line_blocks`` and ``batch.drug_blocks``.
- Baseline predictors use pair identifiers and/or response values only.
- Literature predictors that declare ``requires_raw_feature_datasets`` read the
  raw ``FeatureDataset`` inputs carried on the batch.

Resolving built-in models
-------------------------

Prefer zoo names with ``construct_model`` or ``ModelConfig``:

.. code-block:: python

    from drevalpy.models import construct_model
    from drevalpy.models.config import ModelConfig
    from drevalpy.models.zoo import list_zoo_names
    from drevalpy.types.model_scope import ModelScope

    ElasticNet = construct_model("ElasticNet")
    model = ElasticNet()
    model.build_model(model.get_default_hyperparameters())
    model.train(...)
    model.predict(...)

    # Component-native instance (no DRPModel facade)
    composed = ModelConfig.from_spec("ElasticNet").create_model()

    # Discover presets by scope
    single_drug = list_zoo_names(scope=ModelScope.SINGLE_DRUG)

Named root exports (``ElasticNetModel``, ``DIPKModel``, …) remain available and
are not deprecated.

Programmatic composition
------------------------

For custom stacks without adding a zoo file, pass a recipe as the second argument:

.. code-block:: python

    from drevalpy.models import construct_model
    from drevalpy.models.config import ModelConfig

    CustomElasticNet = construct_model(
        "myElasticNet",
        "scaledGeneExpression:fingerprints:elasticNet",
    )
    model = CustomElasticNet()
    model.build_model({"alpha": 0.1, "l1_ratio": 0.5})

    config = ModelConfig.from_spec("scaledGeneExpression:fingerprints:elasticNet")
    composed = config.create_model()

``construct_model(name)`` / ``construct_model(name, spec)`` return a **class**;
``ModelConfig.create_model()`` returns a **trained-ready** ``ComposedModel`` instance.

Explicit omics view grammar
---------------------------

Cell-line featurizers that operate on a single omics layer use bracket syntax:

.. code-block:: text

    raw[expression]+pca[proteomics]:identity:randomForest

- ``raw[view]`` passes through one dense omics view without preprocessing.
- ``pca[view]`` applies PCA to one dense omics view. The view is required.
- ``+`` concatenates featurizers into ``concatFeaturizers``.

Supported view aliases include ``expression`` (gene expression), ``methylation``,
``mutations``, ``proteomics``, and ``cnv`` (copy-number variation). YAML presets use
the same atoms.

Featurizer hyperparameter tuning (dotted keys)
----------------------------------------------

Ray/Optuna search spaces use **dotted keys** that mirror the composed stack.
Predictor parameters look like ``predictor.<registryName>.<param>``; featurizer
parameters look like ``featurizer.<registry>.<featurizerName>.<index>.<param>``.

Examples for ``ElasticNet`` (``scaledGeneExpression`` + ``fingerprints`` + ``elasticNet``):

.. code-block:: text

    predictor.elasticNet.alpha
    predictor.elasticNet.l1_ratio
    featurizer.cell_line.pca.0.n_components

For ``concatFeaturizers``, each child featurizer gets a zero-based index per name
(``featurizer.cell_line.landmarkGenes.0.standardize``, ``...1.minmax_scale``, …).

Flat ``build_model`` dicts remain supported for predictor keys such as ``alpha`` and
legacy featurizer aliases (``methylation_n_components``). Structured overrides may
also use dotted keys directly.

Previous APIs (through 1.5.1)
-----------------------------

Through version 1.5.1, factory dictionaries and flat view keys were the usual
interfaces. They still work for compatibility but emit ``FutureWarning`` and
should not be used in new code:

- ``MODEL_FACTORY``, ``MULTI_DRUG_MODEL_FACTORY``, ``SINGLE_DRUG_MODEL_FACTORY`` —
  use ``construct_model``, ``ModelConfig.from_spec``, and
  ``list_zoo_names(scope=...)`` instead.
- Flat ``cell_line_views`` / ``drug_views`` in ``build_model`` / hpam YAML —
  configure ``cell_line_featurizer`` / ``drug_featurizer`` in zoo YAML or a recipe
  string instead (see :doc:`example_flexible_inputs`).

Feature contracts and validation limits
---------------------------------------

Each featurizer declares a ``FeatureKind`` (``dense``, ``graph``, or ``sequence``)
via its registration decorator. Predictors declare ``cell_line_contract`` and
``drug_contract``. ``ModelConfig`` validation checks that featurizer output kinds
match predictor input kinds.

``FeatureContract`` currently compares **only the broad ``FeatureKind``**. Graph
compatibility is therefore ``graph`` expected and ``graph`` provided — finer
details such as node feature dimension or edge semantics are **not** validated yet.
Extend contracts carefully when pairing new featurizers with structured predictors.

Scope and early stopping
------------------------

- Multi-drug is the default model scope and requires a drug featurizer for
  feature-based predictors.
- Single-drug models set ``scope: single_drug`` in their zoo YAML; the facade
  exposes ``is_single_drug_model`` for experiment routing.
- Early stopping is derived from predictor capability metadata
  (``supports_early_stopping``) via the zoo predictor name.

Persistence
-----------

Native checkpoints store a versioned payload with the resolved ``ModelConfig``
and fitted component state (``composed_model.joblib``). Legacy checkpoint formats
and deep model import paths are unsupported; see :doc:`runyourmodel`.

Extension path
--------------

Register featurizers/predictors, compose a ``ModelConfig`` or zoo YAML, and use
``construct_model`` / ``load_extensions``. Direct ``DRPModel`` subclass authoring is
not the supported extension mechanism. See :doc:`runyourmodel` for a full example.
