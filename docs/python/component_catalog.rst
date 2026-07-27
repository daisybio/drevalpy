Component catalog
=================

Built-in featurizers and predictors are registered by name. Use these names in
recipe strings, zoo YAML, and ``ModelConfig``. To discover what is loaded at
runtime (including extensions), inspect the registries after import; to add
external components, call ``load_extensions`` (see :doc:`custom_models`).

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
     - No cell-line features (placeholder for baselines that ignore them)
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
     - No drug features (single-drug / naive stacks)

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

load_extensions
---------------

Register external components and optional zoo files before constructing
models:

.. code-block:: python

   from drevalpy.components import load_extensions

   load_extensions(
       directories=["my_components"],
       zoo_files=["my_zoo/toy.yaml"],
   )

See :doc:`custom_models` for a full registration example and
:doc:`architecture` for contracts and composition rules.

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
