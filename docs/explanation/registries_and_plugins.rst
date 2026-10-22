How registries and plugins fit together
=======================================

Whenever you write a name such as ``elasticNet`` in a recipe, ``LSO`` as a split
mode, or ``GDSC2`` as a dataset, DrEvalPy has to find the thing that name stands
for. Registries do that lookup. This page explains what each registry holds, how
they get filled, and how your own code joins them. Reference tables of the
current entries are generated in :doc:`/reference/components`, and the steps for
adding your own are in :doc:`/how-to/extend/index`.

One idea, five registries
-------------------------

A registry maps a human-readable name to an implementation and to a record of
facts about it, such as its description. DrEvalPy has five, one for each concept
that can be extended:

**Predictors.**
   Predictor classes. Each declares the feature format it expects from the
   sample and from the drug side, and its input interface (feature-free, matrix
   or block). :doc:`models_from_components` explains these contracts.

**Featurizers.**
   Featurizer classes, kept separately for the sample side and the drug side.
   Each declares the feature format it produces. A featurizer that works for
   both sides is written once and registered for each side.

**Splitters.**
   One splitter per split mode, registered under its mode name. Each declares the
   leakage rule it enforces, and after every split the registry validates that
   the rule holds. Old names can be registered as aliases of a mode and resolve
   to the canonical name.

**Datasets.**
   Named datasets and the sources they are downloaded from.

**Visualizations.**
   Plot classes that turn results into figures. Each declares requirements, for
   example that the experiment has several models or several folds, or includes
   randomization or robustness tests. The report asks the registry which plots
   apply to an experiment and renders those.

Because models are descriptions made of names (see :doc:`models_from_components`),
the registries are what turns a description into running code. They are also why
a model, a split mode or a plot you register becomes available to the command
line, the Python interface and the report without any change to them.

What every registry offers
~~~~~~~~~~~~~~~~~~~~~~~~~~

The registries share a vocabulary: register an entry, get one by name, list the
names, return one entry's record, and return a summary table. The dataset
registry names its operations for its two kinds of entry (below), and the
predictor and featurizer registries additionally expose class-level helpers to
list names and metadata and to build a table. The method names for each registry
are listed in :doc:`/reference/python_api`.

Registering a name that already exists raises an error by default. The point is
that one package cannot quietly change another package's behaviour. How you
deliberately replace an entry differs between registries:

- The splitter, visualization and dataset registries accept an explicit
  ``override=True`` when you do mean to replace an entry.
- The predictor and featurizer registries have no override. A second
  registration under the same name raises an error, so a custom predictor needs
  its own name.
- Registering the *same* dataset again is a no-op, and registering a conflicting
  one raises an error. Loading the same extension file twice is also a no-op.

Filling the registries
----------------------

Three routes populate the registries, from most to least built in.

**Built-in components** register when you import ``drevalpy``. The import scans
the component directories and imports every module in them, except modules whose
name starts with an underscore. That is why a new built-in component is a new
file with a registration decorator, and why a helper module that should not
register is given a leading underscore. Nothing keeps a separate list of built-ins
in sync by hand.

**Plugins** are installed Python packages that advertise a module under the
``drevalpy.plugins`` entry point group. Importing ``drevalpy`` also imports those
modules, which runs their registration decorators. Nothing else is needed beyond
installing the package. A plugin that fails to import would otherwise silently
remove everything it declares, so the failure is recorded and can be inspected
afterwards. By default a failing plugin is non-fatal, so that one broken
third-party package cannot make the command line unusable. Setting
``DREVALPY_STRICT_PLUGINS=1`` makes it fatal, which is what a
plugin's own tests want.

**Extension directories** are for local experiments. A directory can hold Python
files, which are imported in sorted order (triggering registration decorators for
any registry), and YAML files. A YAML file is read as a dataset declaration when
it has top-level ``sources`` or ``datasets`` keys, and otherwise as a file of zoo
presets. Directories are not searched recursively. You pass the directory
explicitly, or set ``DREVALPY_EXTENSIONS_DIR`` (several directories separated by
the platform's path separator). An explicitly passed list, even an empty one,
takes precedence over the variable, so code that passes its own list is never
affected by the environment.

Plugins and extension files execute arbitrary code when they load. Install and
point to only what you trust. The formats of the extension files are in
:doc:`/reference/extension_files`, and the environment variables are collected in
:doc:`/reference/environment`.

The dataset registry is different
---------------------------------

The other four registries hold code and are rebuilt every time Python starts. The
dataset registry holds *locations*, and its custom entries persist across
sessions in a ``datasets.json`` file under the directory named by
``DREVALPY_CONFIG_DIR``.

It has two kinds of entry. A *source* is a named base location, optionally with
storage credentials, that serves one or more dataset files. A *dataset* combines
a source with a file name. Registering a source once, for example a private
bucket with credentials, lets you add as many datasets under it as you need.

The built-in source is called ``orakl``, an S3 location. Reading from S3 needs
valid credentials: an anonymous request is rejected. DrEvalPy installs the S3 and
HTTP backends of the ``fsspec`` library, so S3 and HTTP(S) sources work out of the
box, while Google Cloud, Azure or other backends need their own ``fsspec``
extras. Datasets are downloaded to a cache directory (``DREVALPY_CACHE_DIR``). The
datasets themselves are described in :doc:`datasets_provenance`.

How the pieces meet in a run
----------------------------

When you run an experiment, names flow through the registries in this order. The
splitter registry turns the split mode into folds and validates them. The
predictor and featurizer registries turn each model's recipe into fresh
components for every fit (see :doc:`training_lifecycle`). The dataset registry
supplies the data. After the run, the visualization registry decides which plots
are possible for the result, based on what the experiment contains.

Code registrations are not remembered. They exist only in the process that made
them and vanish when it ends. A custom component that must be available every
time, or in every process that works on a run, belongs in a plugin or an
extension directory rather than in an interactive session.
