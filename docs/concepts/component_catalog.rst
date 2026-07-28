Component catalog
=================

This catalog is the vocabulary of a DrEvalPy model. Every model is assembled
from components with three distinct roles:

- a **cell-line featurizer** represents the biological sample,
- a **drug featurizer** represents the compound, and
- a **predictor** maps those representations to a drug-response estimate.

The names below are the stable registry names used in recipes and model-zoo
definitions. They are case-sensitive. At this stage, focus on what each
component contributes; the next page, :doc:`from_components_to_models`,
explains how the names fit together and how compatibility is checked.

Cell-line featurizers
---------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Name
     - Role
   * - ``scaledGeneExpression``
     - Landmark gene expression with arcsinh transform and scaling
   * - ``normalizedProteomics``
     - Proteomics with missingness filters and normalization
   * - ``pca``
     - PCA on a dense omics view (``pca[view]``)
   * - ``raw``
     - Pass-through dense view (``raw[view]``)
   * - ``concatFeaturizers``
     - Concatenate several cell-line featurizers (recipe ``+``)
   * - ``landmarkGenes``
     - Landmark gene subset from expression
   * - ``landmarkGenesReduced``
     - Reduced landmark gene subset
   * - ``pathways``
     - Pathway-level cell-line features
   * - ``bionic``
     - BIONIC embeddings for cell lines
   * - ``identity``
     - One-hot encoding of cell-line entity identifiers
   * - ``constant``
     - Constant one-column intercept (no cell-line identity)
   * - ``tissue``
     - Tissue label encoding for naive / tissue-aware models

Drug featurizers
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Name
     - Role
   * - ``concatFeaturizers``
     - Concatenate several drug featurizers (recipe ``+``)
   * - ``view``
     - Pass-through named drug feature view
   * - ``fingerprints``
     - Molecular fingerprints
   * - ``drugGraph``
     - Graph representation of the molecule
   * - ``molgnet``
     - MolGNet embeddings
   * - ``bpePharmaformer``
     - BPE tokenization features for PharmaFormer
   * - ``smilesvec``
     - SMILES vector embeddings
   * - ``identity``
     - One-hot encoding of drug entity identifiers
   * - ``constant``
     - Constant one-column intercept (no drug identity)

Predictors
----------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Name
     - Role
   * - ``naiveMean``
     - Global training-mean baseline
   * - ``naiveDrugMean``
     - Per-drug mean baseline
   * - ``naiveCellLineMean``
     - Per-cell-line mean baseline
   * - ``naiveTissueMean``
     - Per-tissue mean baseline
   * - ``naiveTissueDrugMean``
     - Per-tissue–drug mean baseline
   * - ``naiveMeanEffects``
     - ANOVA-style mean-effects baseline
   * - ``elasticNet``
     - Elastic net on dense features
   * - ``singleDrugElasticNet``
     - Single-drug elastic net
   * - ``lasso``
     - Lasso regression
   * - ``ridge``
     - Ridge regression
   * - ``randomForest``
     - Random forest regressor
   * - ``singleDrugRandomForest``
     - Single-drug random forest
   * - ``svr``
     - Support vector regression
   * - ``gradientBoosting``
     - Gradient boosting regressor
   * - ``adaboost``
     - AdaBoost decision-tree regressor
   * - ``knn``
     - k-nearest neighbors regressor
   * - ``xgboost``
     - XGBoost regressor
   * - ``lightgbm``
     - LightGBM regressor
   * - ``neuralNetwork``
     - Feed-forward neural network
   * - ``drugGNN``
     - Drug graph neural network (literature)
   * - ``precily``
     - Precily (literature)
   * - ``srmf``
     - SRMF (literature)
   * - ``molir``
     - MOLIR (literature)
   * - ``superfeltr``
     - SuperFELTR (literature)
   * - ``pharmaFormer``
     - PharmaFormer (literature)
   * - ``dipk``
     - DIPK (literature)
   * - ``sparsego``
     - SparseGO (literature)

From catalog to composition
---------------------------

A row from each table gives the ingredients for a model, but not yet the model
itself. For example:

- ``scaledGeneExpression`` represents cell lines,
- ``fingerprints`` represents drugs, and
- ``elasticNet`` predicts responses from those representations.

The next page turns those three names into the recipe
``scaledGeneExpression:fingerprints:elasticNet``. It also introduces omics-view
selectors, multi-view concatenation, compatibility checks, and component
hyperparameters. Continue with :doc:`from_components_to_models`; the remaining
sections on this page are reference notes for extensions and older interfaces.

Extensions
----------

External components and optional zoo files can be registered before models are
constructed (via ``load_extensions`` in the Python API). See
:doc:`/python/custom_models` for a full registration example and
:doc:`/python/architecture` for contracts and composition rules.

Backward compatibility
----------------------

Registry names
~~~~~~~~~~~~~~

Before 1.6.0, many of these atoms were reached only through factory dict
entries or deep model modules. Registry string names
(``elasticNet``, ``fingerprints``, …) are the stable composition surface.
Factory dictionaries remain available for backward compatibility, but are
deprecated and may be removed in a future release — prefer
``construct_model`` and the names in the tables above.
