Environment variables
=====================

DrEvalPy reads seven environment variables. All are optional.

.. list-table::
   :header-rows: 1
   :widths: 32 30 38

   * - Variable
     - Default
     - Effect
   * - ``DREVALPY_ARTIFACTS_URI``
     - ``s3://orakl-open-source-data/drevalpy/artifacts/``
     - Location of downloadable model artifacts
   * - ``DREVALPY_ARTIFACTS_STORAGE_OPTIONS``
     - ``{}``
     - fsspec options for the artifact location
   * - ``DREVALPY_CACHE_DIR``
     - ``platformdirs.user_cache_dir("drevalpy")``
     - Where downloaded datasets and artifacts are cached
   * - ``DREVALPY_CONFIG_DIR``
     - ``platformdirs.user_config_dir("drevalpy")``
     - Where user-registered datasets are stored
   * - ``DREVALPY_EXTENSIONS_DIR``
     - unset
     - Extension directories loaded without ``-e``
   * - ``DREVALPY_STRICT_PLUGINS``
     - unset
     - Make a failing plugin an error
   * - ``DREVAL_AE_CACHE``
     - unset
     - Directory that caches trained gene autoencoders

``DREVALPY_ARTIFACTS_URI``
--------------------------

The base URI that DrEvalPy downloads external model artifacts from, such as embeddings and
checkpoints. It can point at a mirror, a local directory or any other fsspec protocol. The
artifact ``<name>`` is cached at ``<cache dir>/artifacts/<name>``. An empty or whitespace-only
value falls back to the default.

``DREVALPY_ARTIFACTS_STORAGE_OPTIONS``
--------------------------------------

A JSON object passed to fsspec when DrEvalPy opens ``DREVALPY_ARTIFACTS_URI``, for example:

.. code-block:: bash

   export DREVALPY_ARTIFACTS_STORAGE_OPTIONS='{"anon": true}'

A value that is not valid JSON, or that is JSON but not an object, logs a warning and is
ignored.

``DREVALPY_CACHE_DIR``
----------------------

The cache directory. It holds one ``<name>.h5mu`` file per downloaded dataset, the
``artifacts/<name>`` entries and ``msigdb/c2.cp.v6.1.symbols.gmt``, the MSigDB gene sets that
Precily downloads on first use when a dataset has no ``uns["pathways_gmt"]``. :doc:`datasets`
describes how ``load`` uses it, including how a cached dataset whose registry entry has a
``sha256`` is replaced when its hash differs. The gene-set file is checked against a pinned
SHA-256 in the same way.

``DREVALPY_CONFIG_DIR``
-----------------------

The configuration directory. It holds ``datasets.json``, the file that stores sources and
datasets registered by ``register_source`` and ``register_dataset``, and a ``.lock`` file
used while that file is written. See :doc:`extension_files` for the registration forms.

``DREVALPY_EXTENSIONS_DIR``
---------------------------

One extension directory, or several joined with ``os.pathsep`` (``:`` on POSIX, ``;`` on
Windows). Empty entries are dropped.

- ``drevalpy.registry.load_extensions()`` reads the variable when ``directories`` is
  ``None``. Any explicit list, including ``[]``, leaves it unread.
- The ``drevalpy`` command loads the variable's directories first, then the directories given
  with ``--extensions-dir`` / ``-e``.

See :doc:`extension_files` for what a directory may contain.

``DREVALPY_STRICT_PLUGINS``
---------------------------

Set to ``1``, ``true``, ``yes`` or ``on`` (case-insensitive) to re-raise the error of a
plugin that fails to import. Importing ``drevalpy.registry`` then raises the plugin's
exception. Without the variable, the failure is logged as a warning and recorded in
``drevalpy.registry.get_failed_plugins()``.

With the variable set, ``drevalpy list plugins`` prints the traceback of the failed import and
exits with status 1. Without it, the ``--strict`` flag of that command exits with status 1
when any plugin failed to load.

``DREVAL_AE_CACHE``
-------------------

A directory for the autoencoders that the ``aeGeneExpression`` sample featurizer trains, as in
the ``RandomForestAEMolGNet`` zoo preset. The ``autoencoder_cache_dir`` option of the
featurizer takes precedence. Without either, nothing is cached and every fit trains its own
autoencoder.

- The first fit trains the autoencoder and writes ``gene_ae_<digest>.pt``. A later fit with
  the same key reads the file instead of training. A file is written to a temporary name first
  and then renamed, so models that run in parallel never read a half-written file.
- The ``<digest>`` covers the autoencoder settings (gene list, epochs, batch size, patience,
  learning rate), the training and early-stopping sample ids, the gene order and the expression
  values. Another fold, gene list, setting or dataset therefore gets its own file.
- The cache stores one training run, so two fits that share a file also share its random
  initialization. Delete the file to train another autoencoder.

The directory is not cleaned up. Remove its files when they are no longer needed.
