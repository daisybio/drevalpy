Model architecture overview
===========================

DrEvalPy now has two cooperating layers:

1. **Component stack** under ``drevalpy.components`` with featurizers, predictors,
   registries, and tuning helpers.
2. **Public orchestration** under ``drevalpy.models`` with ``ModelConfig``, zoo YAML,
   ``ComposedModel``, ``MODEL_FACTORY``, and legacy ``DRPModel`` adapters.

Typical composition:

.. code-block:: text

    ModelConfig -> featurizer(s) + predictor -> ComposedModel

Legacy callers can still use:

.. code-block:: python

    from drevalpy.models import MODEL_FACTORY

    model = MODEL_FACTORY["ElasticNet"]()
    model.build_model(hyperparameters)
    model.train(...)
    model.predict(...)

Programmatic composition is available through:

.. code-block:: python

    from drevalpy.models import construct_model

    model = construct_model("myModel", "scaledGeneExpression:fingerprints:elasticNet")

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
the same atoms:

.. code-block:: yaml

    cell_line_featurizer:
      - raw[expression]
      - pca[methylation]:
          n_components: 100

``construct_model()`` is currently programmatic-only; the CLI still resolves models
through ``MODEL_FACTORY``.

Compatibility notes
-------------------

- Old import paths under ``drevalpy.models.*`` remain as thin re-exports.
- Multi-view sklearn models are zoo presets backed by concat featurizers.
- Hyperparameter grids in YAML were replaced by component-owned defaults and
  structured search spaces. See :doc:`hyperparameter_migration`.
- ``pydantic`` and ``optuna`` are required dependencies for config validation and HPO.

Component-native state
----------------------

Fitted model state now lives in the component stack (``component_stack.joblib`` plus
``hyperparameters.json``). Legacy wrapper attributes such as ``model``, ``drug_means``,
or ``gene_expression_scaler`` are read-only views into that stack for backward
compatibility. Legacy checkpoint formats (``naive_model.json``, ``model.pkl``/``scaler.pkl``,
and literature-specific artifacts) can still be loaded during the deprecation window via
``drevalpy.models._legacy_checkpoint_loaders`` and converted with
``drevalpy.models.legacy_checkpoint_migration.migrate_checkpoint_to_component_stack``.
