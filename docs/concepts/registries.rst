Registries & Extensibility
==========================

Registries are the discovery layer of DrEvalPy. Every extensible concept in the
framework -- predictors, featurizers, splitters, datasets, and visualizations
-- is managed by a registry that maps human-readable names to implementations.

When you write ``elasticNet`` in a recipe, zoo YAML, or CLI invocation,
DrEvalPy resolves that string through the predictor registry to find the class
that implements elastic-net training and prediction. The same principle applies
to featurizers, splitter modes, dataset sources, and plot types.

Common interface
----------------

Every registry exposes the same core operations:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operation
     - Purpose
   * - ``register``
     - Add a new entry (decorator or function call)
   * - ``list``
     - Retrieve a sorted list of all registered names
   * - ``get``
     - Look up an implementation by name
   * - ``metadata``
     - Return one entry's record as a dict
   * - ``table``
     - Return a summary ``DataFrame`` (useful in notebooks)

Registering a name that is already taken **raises** in every registry. Each
``register`` takes an ``override=True`` escape hatch for the cases where
replacing an entry is the intent; the point of the default is that one package
cannot quietly change another package's semantics.

All registries are populated automatically when the package is imported.
Third-party plugins installed via pip are discovered at the same time
(see `Plugin discovery`_ below).

Predictor registry
------------------

Stores predictor classes that map featurized cell-line and drug representations
to a drug-response estimate. Each predictor declares:

- a **cell-line contract** -- what feature format it expects from the cell-line
  featurizer (numeric matrix, graph, or ragged sequence)
- a **drug contract** -- what feature format it expects from the drug featurizer
- an **input interface** -- whether it is feature-free, matrix-based, or
  block-based (see :doc:`component_catalog` for details)

Cell-line featurizer registry
-----------------------------

Stores featurizer classes that transform raw cell-line data (gene expression,
methylation, mutations, CNV, proteomics) into a feature representation
consumable by predictors. Each featurizer declares a **contract** describing
its output format.

Drug featurizer registry
------------------------

Stores featurizer classes that transform raw drug data (SMILES strings,
molecular structures, precomputed embeddings) into feature representations.
Like cell-line featurizers, each drug featurizer declares an output contract.

Splitter registry
-----------------

Stores splitting strategies that divide datasets into train, validation, and
test folds. Each splitter is registered under a mode name (such as ``LPO``,
``LCO``, ``LDO``, or ``LTO``) and declares which **leakage constraint** to
enforce. After every split, the registry automatically validates that the
chosen constraint holds.

Dataset registry
----------------

Manages named datasets and their remote (or local) sources. Unlike the other
registries, dataset entries are persisted to a local configuration file so that
custom datasets survive across sessions.

The registry tracks two kinds of entries:

- **Sources** -- a named base URL that serves as the prefix for one or more
  dataset files. A source can point to any location that
  `fsspec <https://filesystem-spec.readthedocs.io/>`_ supports: public HTTPS
  servers, Amazon S3 buckets, Google Cloud Storage, Azure Blob Storage, local
  file paths, or any other protocol with an fsspec implementation. Optional
  storage credentials (tokens, keys) are stored alongside the URL.
- **Datasets** -- a named reference that combines a source with a filename.
  For example, a dataset named ``GDSC2`` might reference the built-in HTTPS
  source and the file ``GDSC2.h5mu``. When loading, DrEvalPy resolves the full
  path by joining the source URL with the filename.

This two-level design means you register a source once (for instance, a private
S3 bucket with credentials) and then add as many dataset entries under it as
you need -- each pointing to a different ``.h5mu`` file at that location.

Visualization registry
----------------------

Stores visualization classes that generate plots from experiment results.
Each visualization declares **requirements** -- conditions that must be met by
the experiment result for the plot to be applicable (for example: multiple CV
folds, multiple models, or a reference model). The reporting system
automatically selects which visualizations to render based on these
requirements.

Plugin discovery
----------------

When the package is imported, it scans for installed Python packages that
advertise the ``drevalpy.plugins`` entry point group. Importing the advertised
module triggers registration decorators, making a plugin's components
available without any explicit user action beyond installation.

A plugin that raises while importing would otherwise remove every component it
declares without a trace, so the failure is recorded rather than swallowed and
can be read back afterwards. The default stays non-fatal, so one broken
third-party package cannot take the CLI down with it; an environment variable
makes it fatal for the plugin's own CI, where a plugin that does not load is a
failure rather than a degraded experience.

Extension directories
---------------------

For quick local experimentation, both the CLI and the Python API accept an
**extensions directory** containing ``.py`` and ``.yaml`` files. All Python
files in the directory are imported (triggering registration decorators for
any registry), and all YAML files are loaded as model-zoo presets or dataset
declarations. An environment variable (``DREVALPY_EXTENSIONS_DIR``) provides
the same mechanism without requiring a CLI flag.

For code examples of how to interact with each registry, see
:doc:`/python/extensions` (Python) and :doc:`/cli/experiments` (CLI).
