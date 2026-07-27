Models
======

Resolve built-in and custom stacks with ``construct_model`` and
``ModelConfig``. Named facade classes (``ElasticNetModel``, ``DIPKModel``, …)
remain first-class exports. For composition details see :doc:`architecture`;
for registered atoms see :doc:`component_catalog`.

construct_model and ModelConfig
-------------------------------

.. code-block:: python

   from drevalpy.models import construct_model, ElasticNetModel
   from drevalpy.models.config import ModelConfig
   from drevalpy.models.zoo import list_zoo_names
   from drevalpy.types.model_scope import ModelScope

   ElasticNet = construct_model("ElasticNet")
   model = ElasticNet()
   model.build_model(model.get_default_hyperparameters())

   # Same zoo preset as a ComposedModel instance (no DRPModel facade)
   composed = ModelConfig.from_spec("ElasticNet").create_model()

   # Named root export — not deprecated
   also = ElasticNetModel()
   also.build_model(also.get_default_hyperparameters())

   single_drug = list_zoo_names(scope=ModelScope.SINGLE_DRUG)

``construct_model(name)`` / ``construct_model(name, spec)`` return a **class**.
``ModelConfig.create_model()`` returns a trained-ready ``ComposedModel``
instance.

Custom recipe without a zoo file:

.. code-block:: python

   CustomRF = construct_model(
       "MyRF",
       "scaledGeneExpression:fingerprints:randomForest",
   )
   model = CustomRF()
   model.build_model({"n_estimators": 200})

Discover zoo names with ``list_zoo_names()`` (optionally filter by
``ModelScope``). Recipe grammar and featurizer blocks are covered in
:doc:`model_inputs` and :doc:`architecture`.

Lifecycle
---------

The public ``DRPModel`` / ``NativeDRPModel`` facade exposes:

1. ``build_model(hyperparameters)`` — apply flat defaults or overrides.
2. ``train(...)`` / ``predict(...)`` — fit and score on response + feature
   inputs (the experiment runner calls these per fold).
3. ``save(directory)`` / ``load(directory)`` — native
   ``composed_model.joblib`` checkpoints (see :doc:`persistence`).

For day-to-day benchmarking, prefer ``drug_response_experiment`` over a hand-
rolled train loop (:doc:`experiments`).

Backward compatibility
----------------------

Factory dictionaries
~~~~~~~~~~~~~~~~~~~~

Before 1.6.0, ``MODEL_FACTORY``, ``MULTI_DRUG_MODEL_FACTORY``, and
``SINGLE_DRUG_MODEL_FACTORY`` were the usual lookup. This remains available for
backward compatibility, but is deprecated and may be removed in a future
release. Prefer ``construct_model``, ``ModelConfig.from_spec``, and
``list_zoo_names(scope=...)``.

No longer supported
~~~~~~~~~~~~~~~~~~~

Deep imports such as ``drevalpy.models.DIPK.dipk`` or
``drevalpy.models.baselines.*`` no longer resolve. Import built-in models from
the package root instead:

.. code-block:: python

   from drevalpy.models import DIPKModel, ElasticNetModel, construct_model
