Datasets
========

If you are reading this, we assume you are already familiar with these
concepts:

- :doc:`/concepts/datasets`
- :doc:`/concepts/evaluation`

This page explains how to load built-in and custom datasets, and how to split them for training and evaluation.

Built-in datasets
-----------------

Built-in names are listed in the packaged registry. Use
:func:`~drevalpy.data.list_builtin_datasets` to discover them and
:func:`~drevalpy.data.loader.load_mudataset` to load:

.. code-block:: python

   from drevalpy.data import list_builtin_datasets, load_mudataset

   print(list_builtin_datasets())
   # BeatAML2, CCLE, CTRPv1, CTRPv2, GDSC1, GDSC2, PDX_Bruna, TOYv1, TOYv2

   mudataset = load_mudataset("TOYv1")

Built-in loaders download into the system cache directory on first use (see
:doc:`/getting_started/installation` for ``DREVALPY_CACHE_DIR``). Pass
``measure`` to select the response column (for example ``LN_IC50``, ``AUC``,
``response``). When CurveCurator refitting is enabled for a workflow, measure
names gain a ``_curvecurator`` suffix — see :doc:`/concepts/datasets`.

Custom raw and prefit tables
----------------------------

An unknown ``dataset_name`` is treated as a custom load path. Point
``load_mudataset`` at a ``.h5mu`` file directly:

.. code-block:: python

   mudataset = load_mudataset("/path/to/MyStudy.h5mu")

.. _flexible-inputs:

Custom feature tables
---------------------

Baselines such as Random Forest or Elastic Net default to gene expression and
drug fingerprints, but you can reuse the same models with other cell-line or
drug features. You do **not** need a new model class for each omics type; you
only change how the inputs are built.

Feature choice is part of **model composition** (see :doc:`models` and
:doc:`/concepts/from_components_to_models`), not hyperparameters that HPO can
retune. Declare it with a short **recipe** string or the same composition in
zoo YAML.

Example: custom cell-line CSV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose you have a dense matrix ``mynewdatamodality.csv``. Point a Random Forest
at it with ``raw[view]``:

.. code-block:: python

   from drevalpy.models import config, construct_model

   MyRF = construct_model(
       "MyRF",
       "raw[mynewdatamodality]:fingerprints:randomForest",
   )
   model = MyRF()

   # Same composition via ModelConfig + construct_model
   cfg = config.from_spec(
       "raw[mynewdatamodality]:fingerprints:randomForest"
   )
   MyRF2 = construct_model("MyRF", cfg)

.. important::
   Without a custom loader, the CSV must live at
   ``{cache_dir}/{dataset_name}/mynewdatamodality.csv`` when the view name
   is ``mynewdatamodality``.

The same setup as zoo YAML (this is **not** experiment hpam YAML):

.. code-block:: yaml

   cell_line_featurizer:
     name: raw
     view: mynewdatamodality
   drug_featurizer: fingerprints
   predictor: randomForest

Proteomics with built-in preprocessing uses the ``normalizedProteomics``
featurizer:

.. code-block:: text

   normalizedProteomics:fingerprints:randomForest

or in zoo YAML:

.. code-block:: yaml

   cell_line_featurizer:
     normalizedProteomics:
       feature_threshold: 0.7
       n_features: 1000
       normalization_width: 0.3
       normalization_downshift: 1.8
   drug_featurizer: fingerprints
   predictor: randomForest

.. warning::
   Proteomics normalization must not fit on the full dataset: medians computed
   on all samples leak test information into training. The featurizer fits on
   the training split only (``fit_transform``) and applies the learned transform
   to validation/test (``transform``).

Generic CSV loading uses ``load_generic_csv`` from ``drevalpy.data.feature_tables``.
Depending on whether the featurizer is registered under the cell-line or drug
registry, the index column must be ``CELL_LINE_IDENTIFIER``
(``"cell_line_name"``) or ``DRUG_IDENTIFIER`` (``"pubchem_id"``).

Then run the model class through
:func:`~drevalpy.experiment.mu_experiment` the same way as any
other zoo preset — see :doc:`experiments`. For registering entirely new
featurizers or predictors, see :doc:`custom_models`.

Splits
------

:func:`~drevalpy.experiment.mu_experiment` splits the loaded
``MuDataset`` for you (``test_mode`` of ``LPO``, ``LCO``, ``LTO``,
or ``LDO``). You can also use ``MuDataSplitter`` yourself before a custom
training loop:

.. code-block:: python

   from drevalpy.data import load_mudataset, MuDataSplitter

   mudataset = load_mudataset("TOYv1")
   splitter = MuDataSplitter()
   folds = splitter.split(mudataset, mode="LCO", n_splits=5)

For external split scripts, pass ``external_splitter`` to
``MuDataSplitter.split()`` — see :doc:`experiments`. Split semantics are
documented in :doc:`/concepts/evaluation`.
