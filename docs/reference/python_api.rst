Python API reference
====================

This page lists every public symbol with its signature and docstring. The pages are generated
from the source.

Top-level package
-----------------

``import drevalpy`` exposes the functions and classes below, plus ``__version__``. Everything
else is imported from the subpackages in the next section.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   drevalpy.run
   drevalpy.single
   drevalpy.train
   drevalpy.load
   drevalpy.split
   drevalpy.synthetic
   drevalpy.randomization
   drevalpy.robustness

The root package also re-exports ``ModelSpec`` and ``DRPModel``, documented under
``drevalpy.models``, and the ``drevalpy.registry`` subpackage.

Subpackages
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Contents
   * - ``drevalpy.data``
     - Dataset loading, ``concat``, splitting, synthetic datasets and the ``CurveQuality`` rule
   * - ``drevalpy.types``
     - Shared enums and value objects: datasets, split masks, scopes, results
   * - ``drevalpy.experiment``
     - The ``randomization`` and ``robustness`` stress tests
   * - ``drevalpy.evaluation``
     - ``evaluate``, ``score_predictions`` and the metric tables
   * - ``drevalpy.models``
     - ``ModelSpec``, ``combinations``, ``DRPModel``, model configuration, the zoo and tuning
   * - ``drevalpy.components``
     - Featurizers, predictors and their contracts
   * - ``drevalpy.testing``
     - Synthetic fixtures and conformance checks for plugin test suites
   * - ``drevalpy.utils``
     - Shared helpers, for example ``get_response_transformation``
   * - ``drevalpy.visualization``
     - Plots and ``create_report``
   * - ``drevalpy.plugin``
     - The import surface for third-party plugins

Submodules, classes and functions are generated recursively from the package tree.

.. autosummary::
   :toctree: _autosummary
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

Registries and extension loading
--------------------------------

``drevalpy.registry`` holds the registries for sample featurizers, drug featurizers,
predictors, splitters, visualizations and datasets, the plugin discovery state and the
extension loaders. Importing it discovers the installed ``drevalpy.plugins`` entry points.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   drevalpy.registry.load_extensions
   drevalpy.registry.load_extension_dir
   drevalpy.registry.load_extension_file
   drevalpy.registry.load_extension_module
   drevalpy.registry.discover_plugins
   drevalpy.registry.get_loaded_plugins
   drevalpy.registry.get_failed_plugins
   drevalpy.registry.get_skipped_builtin_modules
   drevalpy.registry.register_builtin_components

The registries are the submodules ``drevalpy.registry.sample_featurizer``,
``drevalpy.registry.drug_featurizer``, ``drevalpy.registry.predictor``,
``drevalpy.registry.splitter``, ``drevalpy.registry.visualization`` and
``drevalpy.registry.dataset``.

.. autosummary::
   :toctree: _autosummary

   drevalpy.registry.sample_featurizer
   drevalpy.registry.drug_featurizer
   drevalpy.registry.predictor
   drevalpy.registry.splitter
   drevalpy.registry.visualization
   drevalpy.registry.dataset

Model zoo
---------

``drevalpy.models.zoo`` looks up zoo presets and registers external ones:
``list_zoo_names``, ``get_zoo_config``, ``zoo_model_config``, ``load_external_zoo_file``,
``register_external_zoo_entry`` and ``clear_external_zoo``. It is generated with
``drevalpy.models``.

Plugin facade
-------------

``drevalpy.plugin`` defines nothing of its own. Every name on it is an alias for a symbol
documented elsewhere on this page, except a few whose defining module is private; those are
documented on the facade page itself. The page is generated without index entries, so a
cross-reference to ``Dataset`` resolves to the class and not to one of its two spellings.

.. autosummary::
   :toctree: _autosummary
   :recursive:
   :template: facade.rst

   drevalpy.plugin
