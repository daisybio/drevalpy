.. _flexible-inputs:

Custom inputs for baseline models
=================================

Baselines such as Random Forest or Elastic Net default to gene expression and
drug fingerprints, but you can reuse the same models with other cell-line or
drug features. You do **not** need a new model class for each omics type; you
only change how the inputs are built.

Those inputs are part of the **model architecture** (chosen when you compose
the model), not hyperparameters that HPO can retune. Declare them with a short
**recipe** string (``cellLineFeaturizer:drugFeaturizer:predictor``) or the
same composition in zoo YAML.

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

Generic CSV loading uses ``load_generic_csv`` from ``drevalpy.data.features``.
Depending on whether the featurizer is registered under the cell-line or drug
registry, the index column must be ``CELL_LINE_IDENTIFIER``
(``"cell_line_name"``) or ``DRUG_IDENTIFIER`` (``"pubchem_id"``).

Then run the model class through ``drug_response_experiment`` the same way
as any other zoo preset — see :doc:`experiments`.

See :doc:`hyperparameter_tuning` and :doc:`architecture` for dotted HPO keys
and full composition details. For registering entirely new featurizers or
predictors, see :doc:`custom_models`.

Backward compatibility
----------------------

Views as hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~

Before 1.6.0, cell-line and drug **views** were treated like hyperparameters.
You could pass ``cell_line_views`` / ``drug_views`` to the constructor or put
them in experiment hpam YAML, and in principle retune which inputs a model
used. This remains available for backward compatibility, but is deprecated and
may be removed in a future release.

Inputs are now a fixed part of the architecture (recipe / zoo featurizer
blocks above). Predictor settings such as ``alpha`` remain tunable; which
omics or drug representation you use does not. The old view keys emit a
``FutureWarning``:

.. code-block:: python

   model = construct_model("RandomForest")(
       {
           "cell_line_views": ["mynewdatamodality"],
           "drug_views": ["fingerprints"],
       }
   )

Same idea in experiment hpam YAML (not zoo YAML):

.. code-block:: yaml

   RandomForest:
     cell_line_views:
       - mynewdatamodality
     drug_views:
       - fingerprints

New code should use the recipe / zoo forms at the top of this page instead.
