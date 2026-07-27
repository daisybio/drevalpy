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

.. automodule:: drevalpy.models.zoo
   :members:
   :undoc-members:
   :show-inheritance:

Built-in models
---------------

Every built-in factory name has a zoo YAML under ``drevalpy/models/zoo/``.
Resolve presets with ``construct_model("ElasticNet")`` or build a
``ModelConfig`` with ``ModelConfig.from_spec("ElasticNet")`` and pass it to
``construct_model(name, config)``.
