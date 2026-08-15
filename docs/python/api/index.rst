Python API reference
====================

Guides first — this page is symbol lookup. Prefer the
:doc:`/python/quickstart` track map and the task pages above when learning
workflows; use the packages below when you need signatures and member lists.

The root ``drevalpy`` package re-exports a few conveniences alongside
``__version__`` — ``load``, ``split``, ``run``, ``single``,
``construct_model``, ``randomization``, ``robustness``, and ``registry``. For
everything else, import from the subpackages in the table.

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
     - ``randomization`` and ``robustness`` stress tests
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
     - :doc:`/python/extensions`
     - :doc:`/concepts/component_catalog`
   * - ``drevalpy.plugin``
     - The single supported import surface for third-party plugins
     - :doc:`/python/extensions`
     - :doc:`/concepts/registries`
   * - ``drevalpy.testing``
     - Synthetic fixtures and conformance checks for plugin test suites
     - :doc:`/python/extensions`
     - —
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
   drevalpy.testing
   drevalpy.utils
   drevalpy.visualization

``drevalpy.plugin`` defines nothing of its own — every name on it is an alias for
a symbol documented above, bar a few whose defining module is private and which
are therefore documented on the facade page itself. Its page is generated without
index entries, so a cross-reference to ``Dataset`` resolves to the class rather
than to one of two equally valid spellings of it.

.. autosummary::
   :toctree: _autosummary
   :recursive:
   :template: facade.rst

   drevalpy.plugin
