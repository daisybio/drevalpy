drevalpy.models package
=======================

Root public surface for drug response models.

.. automodule:: drevalpy.models
   :members: DRPModel, MODEL_FACTORY, MULTI_DRUG_MODEL_FACTORY, SINGLE_DRUG_MODEL_FACTORY, construct_model
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
``DIPKModel``) are generated facades backed by those presets. See
:doc:`runyourmodel` and :doc:`model_architecture`.
