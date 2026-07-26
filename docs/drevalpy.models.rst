drevalpy.models package
=======================

Root public surface for drug response models.

.. automodule:: drevalpy.models
   :members: DRPModel, construct_model
   :undoc-members:
   :show-inheritance:

Orchestration helpers
---------------------

.. automodule:: drevalpy.models.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: drevalpy.models.composed_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: drevalpy.models.zoo
   :members:
   :undoc-members:
   :show-inheritance:

Built-in models
---------------

Every built-in factory name has a zoo YAML under ``drevalpy/models/zoo/``.
Named root exports (for example ``ElasticNetModel``, ``NaivePredictor``,
``DIPKModel``) are generated facades backed by those presets. Prefer
``construct_model("ElasticNet")`` or ``ModelConfig.from_spec("ElasticNet")``.
See :doc:`runyourmodel` and :doc:`model_architecture`.

Previous entry point (through 1.5.1)
------------------------------------

Through version 1.5.1, models were typically looked up via ``MODEL_FACTORY``,
``MULTI_DRUG_MODEL_FACTORY``, and ``SINGLE_DRUG_MODEL_FACTORY``. Those
dictionaries remain importable for compatibility but emit a ``FutureWarning``.
Prefer ``construct_model``, ``list_zoo_names(scope=...)``, and ``ModelConfig``.
