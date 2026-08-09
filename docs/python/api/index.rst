Python API reference
====================

Guides first — this page is symbol lookup. Prefer the
:doc:`/python/quickstart` track map and the task pages above when learning
workflows; use the packages below when you need signatures and member lists.

The root ``drevalpy`` package exports ``__version__`` mainly. Import public
APIs from the subpackages in the table.

.. list-table::
   :header-rows: 1
   :widths: 22 30 24 24

   * - Package
     - Purpose
     - Python guide
     - Concepts
   * - ``drevalpy.data``
     - Load screens and custom response tables
     - :doc:`/python/datasets`
     - :doc:`/concepts/datasets`
   * - ``drevalpy.types``
     - Shared enums and value objects (scopes, literature refs, …)
     - :doc:`/python/models`
     - :doc:`/concepts/from_components_to_models`
   * - ``drevalpy.experiment``
     - ``mu_experiment`` and related runners
     - :doc:`/python/experiments`
     - :doc:`/concepts/evaluation`
   * - ``drevalpy.evaluation``
     - ``evaluate`` and metric helpers
     - :doc:`/python/visualization`
     - :doc:`/concepts/evaluation`
   * - ``drevalpy.models``
     - ``construct_model``, ``ModelConfig``, zoo, save/load
     - :doc:`/python/models`
     - :doc:`/concepts/model_zoo`
   * - ``drevalpy.components``
     - Featurizers, predictors, registries, tuning
     - :doc:`/python/custom_models`
     - :doc:`/concepts/component_catalog`
   * - ``drevalpy.utils``
     - Shared helpers
     - —
     - —
   * - ``drevalpy.visualization``
     - Plots and ``create_report``
     - :doc:`/python/visualization`
     - :doc:`/concepts/evaluation`

Submodules, classes, and functions are generated recursively from the package
tree.

.. autosummary::
   :toctree: _autosummary
   :caption: Packages
   :recursive:

   drevalpy.data
   drevalpy.types
   drevalpy.experiment
   drevalpy.evaluation
   drevalpy.models
   drevalpy.components
   drevalpy.utils
   drevalpy.visualization
