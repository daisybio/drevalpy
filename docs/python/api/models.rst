drevalpy.models package
=======================

Root public surface for drug response models.
See :doc:`/python/models` and :doc:`/python/architecture` for usage and design.

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
