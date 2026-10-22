Extension and zoo file formats
==============================

An extension is a Python file, a YAML file or an installed package that adds components,
zoo presets or datasets to DrEvalPy. This page lists what each kind may contain. The loader
functions are in :doc:`python_api`.

Ways to load an extension
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Mechanism
     - What it loads
   * - ``drevalpy -e DIR`` (repeatable)
     - Every ``*.py`` and ``*.yaml`` file directly inside ``DIR``
   * - ``DREVALPY_EXTENSIONS_DIR``
     - The same directories, without the flag; see :doc:`environment`
   * - ``load_extensions(...)``
     - ``modules=`` (importable module names), ``files=`` (``.py`` files), ``directories=``
       and ``zoo_files=`` (zoo YAML files)
   * - ``drevalpy.plugins`` entry point
     - An installed package, imported when ``drevalpy.registry`` is imported

On the command line, the directories in ``DREVALPY_EXTENSIONS_DIR`` load first, then those
given with ``-e``.

Extension directories
---------------------

``load_extension_dir`` does not descend into subdirectories.

#. It imports the ``*.py`` files in sorted order and skips ``__init__.py``. Importing a file
   runs its ``@register`` decorators.
#. It then loads the ``*.yaml`` files in sorted order. Files ending in ``.yml`` are not read.

A YAML file is a dataset declaration if its top level has a ``sources`` or a ``datasets``
key. Any other YAML file is a zoo file. A YAML file whose top level is not a mapping is
skipped.

If a file fails, the error is raised and the components, splitters and visualizations that
the directory registered are removed again. Zoo presets and datasets are not rolled back.
Loading the same file or directory twice is harmless.

Zoo files
---------

A zoo file defines zoo presets that are available by name in
``drevalpy run``, ``drevalpy single`` and ``ModelSpec.from_name``. It is read by
``drevalpy.models.zoo.load_external_zoo_file``, which a directory load also calls. A zoo file
has one of two shapes.

Single zoo preset
~~~~~~~~~~~~~~~~~

The file has a top-level ``predictor`` key. The zoo name is the ``name`` key if present and
the file stem otherwise.

.. code-block:: yaml

   # my_model.yaml, registered as "my_model"
   sample_featurizer: scaledGeneExpression
   drug_featurizer: fingerprints
   predictor: ridge

Mapping of zoo presets
~~~~~~~~~~~~~~~~~~~~~~

Without a top-level ``predictor`` key, each top-level key is a zoo name and each value is a
mapping holding the configuration of that zoo preset. A ``name`` key inside a value is ignored.

.. code-block:: yaml

   MyLasso:
     sample_featurizer: scaledGeneExpression
     drug_featurizer: fingerprints
     predictor: lasso
   MyKnn:
     sample_featurizer: scaledGeneExpression
     drug_featurizer: fingerprints
     predictor:
       name: knn
       hyperparameter_space:
         n_neighbors: {type: int, low: 3, high: 15, default: 5}

Zoo preset configuration
~~~~~~~~~~~~~~~~~~~~~~~~

A zoo preset configuration has these keys. Any other key is rejected.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Key
     - Value
   * - ``sample_featurizer``
     - Optional. A featurizer node, see :doc:`recipes`
   * - ``drug_featurizer``
     - Optional. A featurizer node
   * - ``predictor``
     - Required. A predictor name, ``{name, hyperparameter_space}`` or a one-key mapping
   * - ``prediction_mode``
     - Optional, ``regression`` by default

:doc:`recipes` describes the featurizer node forms and :doc:`hyperparameters` the
``hyperparameter_space`` entries. A zoo name may not equal a built-in zoo name: that
raises ``ValueError``. The whole file is validated before any zoo preset is registered, so a
file with one invalid zoo preset registers none.

Dataset declaration files
-------------------------

A YAML file with a ``sources`` or a ``datasets`` key registers dataset sources and datasets.

.. code-block:: yaml

   sources:
     lab:
       url: s3://my-bucket/screens
       storage_options:
         anon: true
   datasets:
     MyScreen:
       source: lab
       file: MyScreen.h5mu

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``sources.<name>.url``
     - Required. An fsspec base URL
   * - ``sources.<name>.storage_options``
     - Optional. Mapping passed to fsspec
   * - ``datasets.<name>.source``
     - Required. The name of a source
   * - ``datasets.<name>.file``
     - Required. The file name under the source URL

The Python equivalents are ``register_source(name, base_url, storage_options=None, *,
override=False)`` and ``register_dataset(name, source, file, *, override=False)`` in
``drevalpy.registry.dataset``. Registering an identical entry again does
nothing. Registering a conflicting entry raises ``ValueError`` unless ``override=True``.
Custom registrations are stored in ``<config dir>/datasets.json`` and listed with origin
``custom`` by ``drevalpy list datasets``. The built-in source ``orakl`` and its datasets are
described in :doc:`datasets`.

Entry points
------------

A package registers itself in the ``drevalpy.plugins`` entry-point group:

.. code-block:: toml

   [project.entry-points."drevalpy.plugins"]
   my_package = "my_package.drevalpy_components"

DrEvalPy loads each entry point with ``EntryPoint.load()`` when ``drevalpy.registry`` is
imported. Loading imports the referenced module, whose ``@register`` decorators add the
components. A failure is logged as a warning and kept in ``get_failed_plugins()``;
``DREVALPY_STRICT_PLUGINS`` turns it into an error. ``drevalpy list plugins`` shows the load
status of every installed entry point, and ``get_loaded_plugins()`` maps each loaded entry
point name to its value.
