API
===

Import DrEvalPy using

.. code-block:: python

   import drevalpy as dep

Subpackages
-----------

DrEvalPy consists of three major subpackages (datasets, which also contains the split providers, models, and visualization):

* Datasets
* Models
* Visualization

.. toctree::
   :maxdepth: 3

   drevalpy.datasets
   drevalpy.models
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

Response transformations
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: drevalpy.response_transformation
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline function decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: drevalpy.pipeline_function
   :members:
   :undoc-members:
   :show-inheritance:

Command line interface
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: drevalpy.cli.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: drevalpy.cli_run_cv
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: drevalpy.cli_model_testing
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: drevalpy.cli_preprocess_custom
   :members:
   :undoc-members:
   :show-inheritance:
