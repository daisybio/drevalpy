.. _flexible-inputs:

Custom inputs for baseline models
=================================

If you are reading this, we assume you are already familiar with this
concept:

- :doc:`/concepts/from_components_to_models`

Baselines such as Random Forest or Elastic Net default to gene expression and
drug fingerprints, but you can reuse the same models with other cell-line or
drug features. You do **not** need a new model class for each omics type; you
only change how the inputs are built.

Those inputs are part of the **model architecture** (chosen when you compose
the model), not hyperparameters that HPO can retune. Declare them with a short
**recipe** string or the same composition in zoo YAML — the grammar is in
:doc:`/concepts/from_components_to_models`.

Example: custom cell-line CSV
-----------------------------

Suppose you have a dense matrix ``mynewdatamodality.csv``. Point a Random Forest
at it with ``raw[view]``:

.. code-block:: python

   from drevalpy.models import construct_model
   from drevalpy.models.config import ModelConfig

   MyRF = construct_model(
       "MyRF",
       "raw[mynewdatamodality]:fingerprints:randomForest",
   )
   model = MyRF()

   # Same composition via ModelConfig + construct_model
   config = ModelConfig.from_spec(
       "raw[mynewdatamodality]:fingerprints:randomForest"
   )
   MyRF2 = construct_model("MyRF", config)

.. important::
   Without a custom loader, the CSV must live at
   ``{path_to_data}/{dataset_name}/mynewdatamodality.csv`` when the view name
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
     name: normalizedProteomics
     hyperparameters:
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

Generic CSV loading uses ``load_generic_csv`` from ``drevalpy.datasets.feature_tables``.
Depending on whether the featurizer is registered under the cell-line or drug
registry, the index column must be ``CELL_LINE_IDENTIFIER``
(``"cell_line_name"``) or ``DRUG_IDENTIFIER`` (``"pubchem_id"``).

Then run the model class through
:func:`~drevalpy.experiment.drug_response_experiment` the same way as any
other zoo preset — see :doc:`experiments`.

See :doc:`hyperparameter_tuning` for running search on a fixed stack, and
:doc:`architecture` for ``ModelInputBatch`` contracts. For registering entirely
new featurizers or predictors, see :doc:`custom_models`. Deprecated flat
``cell_line_views`` / ``drug_views`` keys are covered under
:doc:`architecture` migration notes.
