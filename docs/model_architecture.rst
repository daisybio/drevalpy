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

``construct_model()`` is currently programmatic-only; the CLI still resolves models
through ``MODEL_FACTORY``.

Compatibility notes
-------------------

- Old import paths under ``drevalpy.models.*`` remain as thin re-exports.
- Multi-view sklearn models are zoo presets backed by concat featurizers.
- Hyperparameter grids in YAML were replaced by component-owned defaults and
  structured search spaces. See :doc:`hyperparameter_migration`.
- ``pydantic`` and ``optuna`` are required dependencies for config validation and HPO.
