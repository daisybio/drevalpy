API
===

Import DrEvalPy using

.. code-block:: python

   import drevalpy as dep

Subpackages
-----------

DrEvalPy consists of four major subpackages:

* Datasets
* Models (``ModelConfig``, zoo, ``ComposedModel``, ``construct_model``)
* Components (featurizers, predictors, registries — first-class building blocks)
* Visualization

Built-in models are composed from registered components; there is a single
construction path via ``ModelConfig`` / zoo presets / ``construct_model``.
The ``MODEL_FACTORY`` dictionaries remain as a deprecated compatibility catalog.
See :doc:`model_architecture` and :doc:`runyourmodel`.

.. toctree::
   :maxdepth: 3

   drevalpy.datasets
   drevalpy.models
   drevalpy.components
   drevalpy.visualization

Other functions
---------------

Major functions for running the experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: drevalpy.experiment
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation functions
~~~~~~~~~~~~~~~~~~~~

.. automodule:: drevalpy.evaluation
   :members:
   :undoc-members:
   :show-inheritance:

Utility functions
~~~~~~~~~~~~~~~~~

.. automodule:: drevalpy.utils
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline function decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: drevalpy.pipeline_function
   :members:
   :undoc-members:
   :show-inheritance:
