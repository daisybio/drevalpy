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
(an explicit ``null`` uses all genes of the view). Views that are left out of ``gene_lists`` keep the default gene list of the model,
and all genes of a list must be present in the dataset.

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

A Random Forest that accepts multiple cell line views simultaneously (in the shipped ``hyperparameters.yaml`` gene expression and
mutations; methylation and copy number variation are optional). Each view is loaded and preprocessed independently, then all feature
matrices are concatenated before training. Methylation data, if used, is reduced with PCA before concatenation
(``methylation_n_components``, default: 100).

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

Random Forest on DIPK's features
-------------------------------------------------------------

``RandomForestAEMolGNet`` takes two of the three learned inputs of the published
:doc:`DIPK <drevalpy.models.DIPK>` model -- an autoencoder embedding of gene expression and a
MolGNet embedding of the drug molecule -- and feeds them to a ``RandomForestRegressor`` instead of
DIPK's neural network. DIPK's third input, the BIONIC network features, is not used at all.

Why this model exists
^^^^^^^^^^^^^^^^^^^^^

DIPK outperforms a Random Forest baseline, but it changes two things at once: the features become
learned embeddings *and* the predictor becomes a neural network. A benchmark cannot tell from that
comparison which of the two changes carries the improvement. This model holds the features fixed
and swaps only the predictor, so the two effects separate. It is therefore primarily an ablation
of DIPK, and only secondarily a model of its own.

What it does
^^^^^^^^^^^^

1. **Cell lines.** Gene expression is restricted to ``gene_list`` (default
   ``gene_expression_intersection``, 2,270 genes, the same as DIPK) and transformed to
   ``log2(x + 1)``. In every
   fold a ``GeneExpressionEncoder`` -- the same autoencoder class DIPK uses -- is trained from
   scratch on the training cell lines and then encodes all cell lines into 512 latent features.
2. **Drugs.** The node-level MolGNet embeddings shipped with each dataset under
   ``DIPK_features/Drugs/MolGNet_<drug>.csv`` are pooled over the atoms of the molecule. The
   default ``mean_max`` concatenates the per-dimension mean and maximum, giving 1536 features.
3. **Prediction.** Both blocks are concatenated and handed to the Random Forest, exactly like any
   other flexible-input sklearn baseline.

The dataset must therefore carry DIPK's feature files; on a dataset without them the model raises
``FileNotFoundError`` rather than falling back to fingerprints.

Differences from ``DIPK``
^^^^^^^^^^^^^^^^^^^^^^^^^

============================  ==========================================  ==================================
Aspect                        ``DIPK``                                    ``RandomForestAEMolGNet``
============================  ==========================================  ==================================
Predictor                     attention network + fully connected head    ``RandomForestRegressor``
Drug features                 node-level MolGNet, pooled by attention     node-level MolGNet, mean+max pooled
BIONIC network features       used as a third input                       not used
Autoencoder training samples  one row per *response*                      one row per *training cell line*
Autoencoder validation set    ``output_earlystopping`` cell lines         held-out fraction of training lines
Gene expression scale         as loaded                                   ``log2(x + 1)`` by default
============================  ==========================================  ==================================

Both models fit the autoencoder inside each fold and only on training cell lines, so neither leaks
test expression into the encoder.

Autoencoder hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The four parameters DIPK also has keep DIPK's names, so that a configuration reads the same for
both models:

- ``epochs_autoencoder`` (default 2000), ``lr_autoencoder`` (1e-4), ``patience_autoencoder`` (50),
  ``batch_size_autoencoder`` (32).

The parameters that only exist here are prefixed ``autoencoder_``:

- ``autoencoder_log2_transform`` (default ``true``): set to ``false`` for data already on a log
  scale.
- ``autoencoder_validation_fraction`` (0.1): share of training cell lines held out for early
  stopping.
- ``autoencoder_seed`` (42): makes that split deterministic.
- ``autoencoder_cache_dir`` (unset): see below.

``batch_size_autoencoder`` defaults to 32 rather than DIPK's 1024 because the training matrix has
one row per cell line instead of one per response, which is smaller by roughly two orders of
magnitude; the smaller batch keeps the number of gradient steps comparable. ``patience_autoencoder``
of 50 stops training long before epoch 2000 on input of that size.

How the published numbers were produced
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module docstring quotes a paired comparison against ``RandomForest`` on CTRPv2 in
leave-cell-line-out over ten folds. Three points matter for reading those numbers:

- They are **normalised** correlations in DrEval's own scale, normalised against the run's
  ``NaiveMeanEffectsPredictor``, where 0 means "no better than the naive mean-effects model".
  They are not raw correlation coefficients.
- The p-value is a **paired two-sided exact Wilcoxon signed-rank test over the ten per-fold
  values** of that metric, this model against the baseline on identical folds -- not the p-value
  of a correlation test. With ``n = 10`` the smallest p this test can attain is 0.0020.
- The run encoded **11,883 genes** (``tcga_autoencoder_genes_available``), a gene list that is not part of the
  downloadable metadata bundle. The default ``gene_expression_intersection`` has 2,270 genes, so the shipped
  configuration encodes a gene space about five times smaller and will not reproduce the numbers exactly. The three
  autoencoder settings above are the ones that run used. The list can be rebuilt with
  ``drevalpy/datasets/featurizer/create_autoencoder_gene_list.py``, see below. To use it, copy it to
  ``<data_path>/meta/gene_lists/`` and set ``gene_list: tcga_autoencoder_genes_available``.

Gene list of the 11,883-gene run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Motivation.** The default list (``gene_expression_intersection``, 2,270 genes) is small. The 11,883-gene space was
chosen for four reasons:

* The TCGA-pretrained encoder and the encoder trained from scratch on the cell lines must see the *same* feature
  space. With different gene sets, the effect of pretraining and the effect of the gene set would be confounded.
* The space is *deployment-optimal*: every gene is measurable by Ion Torrent RNA-seq (Ion AmpliSeq Transcriptome Human
  Gene Expression Panel), so new samples can be scored with the same model.
* Noise genes (not expressed or constant in the cell lines) are removed, but there is deliberately **no
  top-variance selection**: it would discard driver genes with a low variance. The reduction is left to the
  autoencoder bottleneck. Known landmark and clinical driver genes are protected from the noise filter.
* Every gene must exist in the expression data of both CTRPv2 and GDSC1, so that all datasets share one space.

Measured against the 2,270-gene list (``DIPK`` model): normalised Spearman +0.014, normalised PCC +0.010, RMSE equal,
and the spread over folds drops from 0.043 to 0.021 -- no jump, but no loss, and a larger, deployable gene space.

**Construction.** The list is a chain of set operations; each step writes its own list to
``<data_path>/meta/gene_lists/``. Nothing in the chain uses the pretrained autoencoder or response data:

1. ``ctrpv2_gdsc1_intersection`` (16,159 genes): genes in the expression data of CTRPv2 and GDSC1.
2. ``ctrpv2_gdsc1_iontorrent_intersection`` (15,701): intersected with the genes of the Ion Torrent panel.
3. ``autoencoder_expressed_genes`` (11,884): noise filter on the CTRPv2 expression
   (``log2(x + 1)``): a gene is dropped if its mean is below 0.5 or its variance below 1e-3, unless it is in the
   protect list (``--protect``, ``landmark_plus_clinical_drivers.csv``, 893 genes).
4. ``tcga_autoencoder_genes_available`` (11,883): minus the genes without a locus in the TCGA gene annotation
   (``gencode.v23.probemap.tsv``), which cannot be mapped to TCGA expression. Only SCAND3 is dropped.

**Inputs.** By default the script downloads the gene content of the panel: the public *AmpliSeq for Illumina
Transcriptome RNA Manifest* (dated 2019-03-13, anonymous, 20,802 genes; the ``[Targets]`` section, column ``Gene``)

.. code-block:: text

    https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/ampliseq-for-illumina/ampliseq-for-illumina-transcriptome-rna-manifest-20190313.zip

It has the same genes as the Ion AmpliSeq Transcriptome Human Gene Expression Panel (user guide
`MAN0010742 <https://assets.thermofisher.com/TFS-Assets/LSG/manuals/MAN0010742_AmpliSeqTranscriptomeHumanGeneExpr_UG.pdf>`_,
over 20,000 genes); the 10 ERCC spike-in controls of the Thermo Fisher BED file are not part of it and are not in the
data anyway, so the result is identical. Alternatively, pass the Thermo Fisher target file with ``--panel``
(``hg19_AmpliSeq_Transcriptome_21K_v1.bed``, ERCC variant; semicolon separated, no header, gene symbol in the seventh
column, adjustable with ``--panel_symbol_column``). The protect list and the TCGA probemap are inputs as well.

.. code-block:: bash

    python -m drevalpy.datasets.featurizer.create_autoencoder_gene_list --data_path data \
        --protect <landmark_plus_clinical_drivers.csv> --probemap <gencode.v23.probemap.tsv>

With the original inputs the four lists are reproduced byte-identically.

Reusing an encoder across models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The encoder depends only on the training cell lines, the gene list and the autoencoder
hyperparameters -- not on the drug features. Setting ``autoencoder_cache_dir`` (or the
``DREVAL_AE_CACHE`` environment variable) makes models that share those train the encoder once per
fold and reuse it. Apart from saving time this makes such models pairwise comparable: with an
identical encoder, any difference between them cannot come from the encoder. Caching is off unless
one of the two is set, and the cache key covers the gene *order* as well as the gene names, since
the same genes in a different order produce a different encoder.

.. automodule:: drevalpy.models.baselines.ae_molgnet_random_forest
   :members:
   :undoc-members:
   :show-inheritance:
