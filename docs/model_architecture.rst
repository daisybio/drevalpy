Model architecture overview
===========================

DrEvalPy has two cooperating layers:

1. **Component stack** under ``drevalpy.components`` with featurizers, predictors,
   registries, and tuning helpers.
2. **Public orchestration** under ``drevalpy.models`` with ``ModelConfig``, zoo YAML,
   ``ComposedModel``, and a single generated ``NativeDRPModel`` facade exposed via
   ``MODEL_FACTORY``.

Typical composition:

.. code-block:: text

    ModelConfig -> featurizer(s) + predictor -> ComposedModel
    MODEL_FACTORY[name] -> NativeDRPModel facade -> same ComposedModel stack

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

Public callers continue to use:

.. code-block:: python

    from drevalpy.models import MODEL_FACTORY

    model = MODEL_FACTORY["ElasticNet"]()
    model.build_model(hyperparameters)
    model.train(...)
    model.predict(...)

Programmatic composition:

.. code-block:: python

    from drevalpy.models import construct_model
    from drevalpy.models.config import ModelConfig

    model = construct_model("myModel", "scaledGeneExpression:fingerprints:elasticNet")
    composed = ModelConfig.from_spec("ElasticNet").create_model()

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

Scope and early stopping
------------------------

- Multi-drug is the default model scope and requires a drug featurizer for
  feature-based predictors.
- Single-drug models set ``scope: single_drug`` in their zoo YAML; the facade
  exposes ``is_single_drug_model`` for experiment routing.
- Early stopping is derived from predictor capability metadata
  (``supports_early_stopping``), not from adapter mixins.

Persistence
-----------

Native checkpoints store a versioned payload with the resolved ``ModelConfig``
and fitted component state (``composed_model.joblib``). Legacy checkpoint formats
and deep model import paths are unsupported.

Extension path
--------------

Register featurizers/predictors, compose a ``ModelConfig`` or zoo YAML, and use
``construct_model`` / ``MODEL_FACTORY``. Direct ``DRPModel`` subclass authoring is
not the supported extension mechanism.
