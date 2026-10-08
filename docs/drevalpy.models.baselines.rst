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

For the multi-view models (``MultiViewRandomForest``, ``MultiViewXGBoost``, ``MultiViewLightGBM``), the cell line views and the
gene list of each view are set together in ``view_configs``. Every entry is one configuration that is tried during
hyperparameter tuning:

.. code-block:: yaml

    MultiViewRandomForest:
      view_configs:
        - cell_line_views:
            - gene_expression
            - mutations
          gene_lists:
            gene_expression: landmark_genes
            mutations: drug_target_genes_all_drugs
      drug_views:
        - fingerprints
      ...

Selecting genes
^^^^^^^^^^^^^^^

Gene-based views are restricted to a list of genes, which is stored in ``<data_path>/meta/gene_lists/<gene_list>.csv``
(column ``Symbol``). The single-view models set it with the hyperparameter ``gene_list``
(the shipped ``hyperparameters.yaml`` uses ``landmark_genes``; the code default is ``landmark_genes_reduced``):

.. code-block:: yaml

    RandomForest:
      cell_line_views:
        - gene_expression
      gene_list:
        - landmark_genes
      ...

The multi-view models and MOLIR and SuperFELTR take a dictionary ``gene_lists`` with one gene list per view
(``null`` uses all genes of the view). All genes of the list must be present in the dataset.

How features are loaded
^^^^^^^^^^^^^^^^^^^^^^^

The feature loading depends on which view is specified in the configuration:

- **gene_expression**: Loaded with the gene list set by ``gene_list`` for feature selection.
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
a single input matrix. Available models (names as registered for ``--models``): ``ElasticNet``, ``Lasso``, ``RandomForest``,
``SVR``, ``GradientBoosting``, ``AdaBoostDecisionTree``, and ``KNNRegressor``.

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

A Random Forest that accepts multiple cell line views simultaneously (by default gene expression and mutations; methylation and
copy number variation are optional). Each view is loaded and preprocessed independently, then all feature
matrices are concatenated before training. Methylation data, if used, is reduced with PCA before concatenation.

.. automodule:: drevalpy.models.baselines.multi_view_random_forest
   :members:
   :undoc-members:
   :show-inheritance:

Multi-View XGBoost
-------------------------------------------------------------

An XGBoost regressor that accepts multiple cell line views, configured like the ``MultiViewRandomForest``.

.. automodule:: drevalpy.models.baselines.multi_view_xgboost
   :members:
   :undoc-members:
   :show-inheritance:

Multi-View LightGBM
-------------------------------------------------------------

A LightGBM regressor that accepts multiple cell line views, configured like the ``MultiViewRandomForest``.
It requires the ``lightgbm`` package.

.. automodule:: drevalpy.models.baselines.multi_view_lightgbm
   :members:
   :undoc-members:
   :show-inheritance:
