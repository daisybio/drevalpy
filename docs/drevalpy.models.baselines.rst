Implemented baselines
=================================

.. _flexible-inputs:

Flexible Input System
--------------------------------------------

The sklearn baseline models support **flexible inputs**. Rather than hardcoding which omic data type a model uses,
you configure ``cell_line_views`` and ``drug_views`` directly in the ``hyperparameters.yaml`` file.
A single model class (e.g., ``ElasticNet``, ``RandomForest``, ``KNNRegressor``) can therefore be trained on gene expression,
proteomics, or any other available omic without needing a separate Python class for each combination.

This replaces the previously separate model classes (``ProteomicsRandomForest``, ``ProteomicsElasticNet``,
``SingleDrugProteomicsRandomForest``, ``SingleDrugProteomicsElasticNet``), which have been removed in favor
of this unified approach.

Configuring the input views
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default ``RandomForest`` configuration uses gene expression and fingerprints:

.. code-block:: yaml

    RandomForest:
      cell_line_views:
        - gene_expression
      drug_views:
        - fingerprints
      n_estimators:
        - 100
      max_depth:
        - 5
        - 10
        - 30
      ...

To train the same Random Forest on **proteomics** data instead, change ``cell_line_views``:

.. code-block:: yaml

    RandomForest:
      cell_line_views:
        - proteomics
      drug_views:
        - fingerprints
      n_estimators:
        - 100
      ...

For the ``MultiViewRandomForest``, multiple cell line views can be specified as a nested list:

.. code-block:: yaml

    MultiViewRandomForest:
      cell_line_views:
        - - gene_expression
          - methylation
          - mutations
          - copy_number_variation_gistic
      drug_views:
        - fingerprints
      ...

How features are loaded
^^^^^^^^^^^^^^^^^^^^^^^

The feature loading depends on which view is specified in the configuration:

- **gene_expression**: Loaded with the ``landmark_genes_reduced`` gene list for feature selection.
- **fingerprints**: Loaded using the precomputed Morgan fingerprints provided with each dataset.
- **proteomics**: Loaded as a generic CSV. The ``ProteomicsMedianCenterAndImputeTransformer`` is
  automatically initialized for preprocessing.
- **Any other feature name** (e.g., ``methylation``, ``mutations``, ``copy_number_variation_gistic``,
  or a custom name): The model calls ``load_generic_csv``, which looks for a CSV file at
  ``<data_path>/<dataset_name>/<feature_name>.csv``. The CSV must have ``cell_line_name`` as the index column.
  All columns (except ``cellosaurus_id``, which is dropped if present) are used as features.

This means you can use **any custom omic** by placing a correctly formatted CSV in the dataset directory
and setting ``cell_line_views`` to the file's name (without the ``.csv`` extension).

For drug features the same logic applies: ``fingerprints`` loads the precomputed fingerprints, an empty
``drug_views`` list loads only the drug IDs, and any other name loads the CSV at
``<data_path>/<dataset_name>/<feature_name>.csv``.

Proteomics-specific hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``proteomics`` is specified as a cell line view, the following hyperparameters control the
preprocessing transformer:

- ``proteomics_feature_threshold`` (default: 0.7): minimum fraction of non-NA values required per protein
- ``proteomics_n_features`` (default: 1000): number of top-variance features to select
- ``proteomics_normalization_width`` (default: 0.3): width parameter for median-center normalization
- ``proteomics_normalization_downshift`` (default: 1.8): downshift parameter for median-center normalization

Naive Predictors
--------------------------------------------

Simple mean-based predictors that serve as lower-bound baselines. These models do not use any cell line
or drug features. They predict the mean response value computed from the training set, aggregated at
different levels (global, per drug, per cell line, per tissue, or per tissue-drug combination).

.. automodule:: drevalpy.models.baselines.naive_pred
   :members:
   :undoc-members:
   :show-inheritance:

Sklearn Models
------------------------------------------------

Scikit-learn-based models for drug response prediction. All models in this module support flexible inputs
(see :ref:`flexible-inputs` above). By default they concatenate cell line features and drug features into
a single input matrix. Available models: ``ElasticNetModel``, ``RandomForest``, ``SVMRegressor``,
``GradientBoosting``, and ``AdaBoostDecisionTree``.

.. automodule:: drevalpy.models.baselines.sklearn_models
   :members:
   :undoc-members:
   :show-inheritance:

Single-Drug Baselines
-----------------------------------------------------------

Single-drug variants of the sklearn models. These models are trained separately for each drug, using only
cell line features (no drug features). Available models: ``SingleDrugRandomForest`` and
``SingleDrugElasticNet``. Both support flexible inputs for the cell line view.

.. automodule:: drevalpy.models.baselines.singledrug_baselines
   :members:
   :undoc-members:
   :show-inheritance:

Multi-View Random Forest
-------------------------------------------------------------

A Random Forest that accepts multiple cell line views simultaneously (e.g., gene expression, methylation,
mutations, and copy number variation). Each view is loaded and preprocessed independently, then all feature
matrices are concatenated before training. Methylation data is reduced with PCA before concatenation.

.. automodule:: drevalpy.models.baselines.multi_view_random_forest
   :members:
   :undoc-members:
   :show-inheritance:
