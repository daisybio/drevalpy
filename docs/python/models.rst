Models
======

Resolve built-in and custom stacks with ``construct_model`` and declarative
``ModelConfig``. For composition details see :doc:`architecture`; for registered
atoms see :doc:`component_catalog`.

Resolution path
---------------

The usual flow is:

.. code-block:: text

   ModelConfig (recipe / zoo / dict)
        -> construct_model(name[, spec])
        -> DRPModel subclass
        -> fresh instance: ModelClass(hyperparameters=None)

Day-to-day use is two steps: resolve a **class**, then construct an
**instance**.

.. code-block:: python

   from drevalpy.models import construct_model

   ElasticNet = construct_model("ElasticNet")  # DRPModel subclass
   model = ElasticNet()  # instance with class defaults
   model = ElasticNet({"alpha": 0.1})  # instance with flat overrides

``ModelConfig`` describes the featurizer/predictor stack (from a zoo name,
recipe string, YAML file, or dict). It does **not** build runnable models by
itself — pass it to ``construct_model`` when you already have a config object:

.. code-block:: python

   from drevalpy.models import construct_model
   from drevalpy.models.config import ModelConfig

   config = ModelConfig.from_spec("ElasticNet")
   ElasticNet = construct_model("ElasticNet", config)
   model = ElasticNet()

Discover available zoo names with ``list_zoo_names()`` (optionally filter by
``ModelScope``).

Custom recipe without a zoo file:

.. code-block:: python

   CustomRF = construct_model(
       "MyRF",
       "scaledGeneExpression:fingerprints:randomForest",
   )
   model = CustomRF({"n_estimators": 200})

Recipe grammar and featurizer blocks are covered in :doc:`model_inputs` and
:doc:`architecture`.

Lifecycle
---------

Each ``DRPModel`` subclass exposes:

1. ``ModelClass(hyperparameters=None)`` — construct with class defaults or flat
   overrides. Hyperparameters and view lists are immutable after construction;
   create a new instance to change configuration.
2. ``train(...)`` / ``predict(...)`` — fit and score on response + feature
   inputs (the experiment runner constructs a fresh instance per fold).
3. ``save(directory)`` / ``ModelClass.load(directory)`` — native
   ``model.joblib`` checkpoints (format ``drevalpy-model``; see
   :doc:`persistence`).

Predictors inside a ``ModelConfig`` receive static hyperparameters at
construction (``PredictorConfig.create_instance()``). Dimension-dependent
allocation happens privately during ``fit()``; there is no public
``Predictor.build``.

For day-to-day benchmarking, prefer ``drug_response_experiment`` over a hand-
rolled train loop (:doc:`experiments`).

Backward compatibility
----------------------

Factory dictionaries
~~~~~~~~~~~~~~~~~~~~

Before 1.6.0, ``MODEL_FACTORY``, ``MULTI_DRUG_MODEL_FACTORY``, and
``SINGLE_DRUG_MODEL_FACTORY`` were the usual lookup. They remain as **lazy,
built-in-only** compatibility views equivalent to ``construct_model(name)`` for
zoo preset names, but emit ``FutureWarning`` and may be removed in a future
release. Prefer ``construct_model``, ``ModelConfig.from_spec``, and
``list_zoo_names(scope=...)``.

Named root exports (``ElasticNetModel``, ``DIPKModel``, …) are removed. Use
``construct_model("ElasticNet")`` (or the zoo preset string) instead.

``ModelConfig.create_model()`` is removed. Use
``construct_model(name_or_recipe)()`` or ``construct_model(name, config)()``.

No longer supported
~~~~~~~~~~~~~~~~~~~

Deep imports such as ``drevalpy.models.DIPK.dipk`` or
``drevalpy.models.baselines.*`` no longer resolve. Resolve models with
``construct_model`` from ``drevalpy.models``.

Legacy checkpoint formats (including ``composed_model.joblib``) are not loadable.
Retrain and persist via ``model.save`` / ``ModelClass.load`` (``model.joblib``).
