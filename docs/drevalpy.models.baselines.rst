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

Mutation status of the target gene
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sklearn models can append the mutation status of the gene a drug acts on. The feature depends on the pair of
cell line and drug (BRAF for dabrafenib, PIK3CA for alpelisib), so it is built in ``get_concatenated_features``
and is not a cell line or a drug view. It is switched on with the hyperparameter ``target_mutation``:

.. code-block:: yaml

    RandomForest:
      target_mutation:
        - true
      ...

The block has 6 columns, appended to the right of the other features: a one-hot encoding of the status
(``activating``, ``LoF``, ``other``, ``wildtype``, ``missing``) and ``target_sensitizing``, which is 1 if the status
equals the status that makes cells sensitive to the drug. Drugs without an entry get only zeros, and a cell line
that is not in the status table gets ``missing``. It needs two tables in ``<data_path>/meta/target_mutation/``
(another directory can be set with ``target_mutation_path``):

- ``curated_drug_target_variant.csv``: ``pubchem_id``, ``biomarker_gene`` and ``sensitive_status`` per drug.
- ``functional_status_matrix.csv``: ``cellosaurus_id``, ``gene`` and ``status`` per cell line and gene.

Cell lines are matched through the ``cell_line_names.csv`` of the dataset. The block is standardized with the mean and
standard deviation of the training rows, which linear models with a penalty need. Set
``target_mutation_standardize: false`` to switch it off.

Where the tables come from. All scripts are in ``drevalpy/datasets/featurizer/`` and write to
``<data_path>/meta/target_mutation/``. Run them in this order (``--data_path`` defaults to ``data``):

.. code-block:: bash

    python -m drevalpy.datasets.featurizer.create_target_mutation_status --data_path data
    python -m drevalpy.datasets.featurizer.create_curated_drug_target_table --data_path data
    python -m drevalpy.datasets.featurizer.check_curated_drug_target_table --data_path data  # optional

- ``functional_status_matrix.csv`` is built from the DepMap 24Q4 files ``OmicsSomaticMutations.csv`` and ``Model.csv`` by
  ``drevalpy/datasets/featurizer/create_target_mutation_status.py``, which also downloads them from figshare (md5
  checked). Per variant and gene the status is ``activating`` (hotspot or oncogene high impact), ``LoF`` (truncating
  variant; for the tumour suppressors TP53, PTEN and NF1 also hotspots), ``other`` or ``wildtype``. An oncogene counts as
  ``LoF`` only if all its variants are truncating. ``PI3K_AKT`` and ``RAS_MAPK`` are aggregates of their components.
  A splice variant without protein change is not recognized as truncating, so it can stay ``activating``.
  Without arguments the script uses every ``<data_path>/*/cell_line_names.csv`` and ``mutations.csv``, so the matrix
  covers all datasets in ``data_path``.
- ``curated_drug_target_variant.csv`` is built by ``drevalpy/datasets/featurizer/create_curated_drug_target_table.py``
  from the seed table ``target_mutation_seed.csv`` (next bullet). It does not read any response data, so the table is
  the same for every dataset and the evaluation does not depend on how it was curated. The seed can hold several
  biomarker genes per drug, the model uses one, so the script keeps exactly one row per drug. Candidates are the rows
  whose gene the status matrix can represent; drivers that are fusions or amplifications (ALK, MET, ERBB2, FGFR1,
  FGFR4) and genes outside the matrix are dropped. Among the candidates the row with the best CIViC ``evidence_level``
  wins (A before E, rules without a level last), then the one with the most ``n_evidence_items``, then the first in the
  seed. The script also writes ``curated_drug_target_report.csv`` with the outcome per seed row (``kept``, ``other row
  of the drug selected``, ``not representable (fusion or amplification)`` or ``gene not in the status matrix``) and
  checks the status values. If the seed file given with ``--seed`` (default: the shipped file) does not exist, the
  script builds it first with ``create_target_mutation_seed.py`` (internet needed, about 15 minutes).
- ``target_mutation_seed.csv`` links each drug to a biomarker gene and the alteration state associated with
  sensitivity. It is derived from public databases by ``create_target_mutation_seed.py``, see
  :ref:`target-mutation-seed` below. Drugs that a dataset lists under a code name and that have no entry in the
  databases get no row and therefore only zeros in the block.
- ``check_curated_drug_target_table.py`` is an optional diagnostic. For every drug it compares the AUC of the
  sensitizing group with all other cell lines (Cohen d, Mann-Whitney) in the given datasets and writes
  ``curated_drug_target_check.csv``. It changes nothing; it shows where an entry is not supported by the data.

.. _target-mutation-seed:

Creating the seed table
~~~~~~~~~~~~~~~~~~~~~~~

``create_target_mutation_seed.py`` creates ``target_mutation_seed.csv``. The script needs no input table: it reads the
drug lists of the datasets that drevalpy provides and derives targets and biomarkers from public databases and
literature at run time. No per-drug rule is supplied to the script, and no drug-response data is read.

.. code-block:: bash

    python -m drevalpy.datasets.featurizer.create_target_mutation_seed --data_path data --out target_mutation_seed.csv

- ``--data_path`` (default ``data``): drevalpy data directory; ``<data_path>/<dataset>/drug_names.csv`` is used if
  present.
- ``--datasets`` (default: all drevalpy datasets): datasets to use, e.g. ``--datasets GDSC1 CTRPv2``.
- ``--out`` (default ``target_mutation_seed.csv``): output file.
- ``--no_remote``: do not fetch datasets that are not in ``--data_path``.
- ``--min_pchembl`` (default ``7.0``): potency cutoff of the binding-activity target fallback (step 3).
- ``--workers`` (default ``3``): parallel target look-ups.
- ``--max_drugs``: process only the first N drugs (testing).
- ``--email``: optional contact address passed to Europe PMC.

It needs internet access to ``civicdb.org``, ``pubchem.ncbi.nlm.nih.gov``, ``www.ebi.ac.uk`` (ChEMBL, Europe PMC) and,
for datasets that are not stored locally, ``zenodo.org``. For all 9 datasets (about 1,190 distinct drugs) it runs for
roughly 15 minutes, dominated by the rate-limited PubChem and ChEMBL requests.

What the script does:

1. **Collects the drugs.** For every dataset in ``AVAILABLE_DATASETS`` (the derived ``CTRPv2_clean*`` variants are
   skipped) the file ``drug_names.csv`` is read. A dataset that is not in ``--data_path`` is not downloaded: only that
   one file is fetched from its Zenodo archive by HTTP range requests (for the TOY datasets from the nf-core test-data
   repository). Drugs are merged by ``pubchem_id``, so the ``pubchem_id`` of the output is the one of the datasets.
2. **Pulls all predictive evidence from CIViC** (GraphQL): every ``PREDICTIVE`` evidence item with significance
   ``SENSITIVITYRESPONSE`` or ``RESISTANCE``, linking a therapy to a molecular profile (gene and variant) with an
   evidence level and a PubMed ID. **Reviews are not used as evidence:** the publication type of every cited PubMed ID
   is read from Europe PMC, and an item whose source is a review is skipped (counted in ``review_items_excluded``).
   Sources without a PubMed ID (a DOI or conference abstract) cannot be typed and are kept.
3. **Finds the target genes of the drug.** The ChEMBL molecule is found via the PubChem CID (for drugs without a
   numeric CID by exact name). ``chembl_mechanism`` uses the targets of the mechanism-of-action records of the molecule
   and of its parent. Only if there is no such record, ``chembl_binding_activity`` uses the human targets with potent
   binding-assay activity (pChEMBL >= ``--min_pchembl``) and at least half as many measurements as the most-measured
   target.
4. **Matches the drug to a CIViC therapy** by any of its dataset names, then by PubChem and ChEMBL synonyms (for
   example MK-1775 and adavosertib). A name that differs from ``drug_name`` is recorded in ``source``.
5. **Direct rules.** Every target gene with CIViC sensitivity evidence for the drug yields one row. If none has, one
   gene of the same pathway bucket (``RAS_MAPK``, ``PI3K_AKT``) as a target is used. Evidence on other genes
   (combination partners, context) is ignored. For fusion profiles ``A::B`` only the 3' gene counts.
6. **Indirect rules** (synthetic-lethal or dependence biomarkers that are not the target): PARP1/PARP2 inhibitors give
   BRCA1, BRCA2, PALB2, ATM, RAD51C, RAD51D or BARD1 ``LoF``; MDM2/MDM4 inhibitors give TP53 ``wildtype``;
   ATR/CHEK1/CHEK2/WEE1 inhibitors give TP53 ``LoF``. A rule is only written if it is confirmed: either CIViC has
   evidence for the drug on that gene in the expected direction (``civic_predictive_indirect``), or, for compounds that
   CIViC does not list, Europe PMC returns a primary paper that names the compound in its title
   (``mechanism_literature``). Reviews and drug-screen papers (GDSC, CCLE) are rejected by title pattern.
7. **Classifies the status** from the CIViC profile text, aware of the gene role: ``wildtype`` stays ``wildtype``;
   loss of function, loss and inactivating give ``LoF``, and so does a plain mutation, deletion, frameshift or
   truncation of a canonical tumour suppressor; everything else is ``activating``.
8. **Writes the table** and prints a summary, including a warning if web requests still failed after retries (run
   the script again to complete the affected drugs).

The script embeds three small class-level tables, not per-drug rules: ``PATHWAY``, ``TSG`` (tumour suppressors) and
``INDIRECT_MAP`` (step 6).

Columns of the seed:

========================= ===========================================================================================
column                    description
========================= ===========================================================================================
``drug_name``             first name found in the datasets.
``pubchem_id``            PubChem CID as used in the drevalpy datasets.
``datasets``              drevalpy datasets containing the drug, separated by ``;``.
``biomarker_gene``        gene whose alteration state is associated with sensitivity.
``sensitive_status``      ``activating``, ``LoF`` or ``wildtype``.
``target_genes``          target genes from ChEMBL, separated by ``;``.
``target_source``         ``chembl_mechanism`` or ``chembl_binding_activity``.
``evidence_type``         ``civic_predictive_target``, ``civic_predictive_pathway``, ``civic_predictive_indirect`` or
                          ``mechanism_literature`` (steps 5 and 6).
``evidence_level``        best CIViC level of the evidence used: A validated, B clinical, C case study, D preclinical,
                          E inferential; empty for ``mechanism_literature``.
``n_evidence_items``      number of CIViC evidence items used for this drug and gene (1 for ``mechanism_literature``).
``review_items_excluded`` number of evidence items not used because their source is a review.
``source``                the evidence: CIViC evidence IDs (at most five), level and PubMed IDs, or the paper that
                          confirms a ``mechanism_literature`` rule.
========================= ===========================================================================================

The shipped seed was created on 2026-10-23 from all 9 datasets: 186 rules for 115 drugs (150 target, 5 pathway, 19
indirect, 12 mechanism-literature; 151 ``activating``, 34 ``LoF``, 1 ``wildtype``). Of the 174 CIViC-based rules 42
have best level A, 48 B, 23 C and 61 D or E. After the choice of one row per drug and the removal of genes that the
status matrix cannot represent, 66 drugs remain in the curated table.

Limitations:

- The result depends on the live databases and changes with their releases; record the run date with the output.
- Only drugs with evidence receive a rule: 606 of the 1,190 drugs have no target in ChEMBL (mostly combination
  preparations and internal codes), and a rule needs CIViC evidence on a target gene or an indirect class.
- ChEMBL records some targets at family or complex level (FKBP1A for the mTOR inhibitors sirolimus and temsirolimus,
  which therefore get no MTOR rule). The PI3Kbeta to PTEN class is not derived because it holds for beta-selective
  inhibitors only. Nutlin-3 has an MDM2 target, but CIViC does not list it and its discovery paper does not name it in
  the title, so it gets no rule.
- ``mechanism_literature`` rules rest on a mechanism principle; the cited paper characterizes the compound but does
  not test the biomarker in the listed direction. Filter on ``evidence_level`` where only well-supported rules are
  wanted.
- Screens are excluded by title only. A check of the cited PubMed IDs found no large pharmacogenomic screen (GDSC,
  CCLE, CTRP), but a paper whose title gives no hint that it re-analyses such data would not be detected.
- Synonym matching uses PubChem and ChEMBL name synonyms; the matched name is recorded in ``source``.

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
cell line features (no drug features). Available models: ``SingleDrugRandomForest``,
``SingleDrugElasticNet`` and ``SingleDrugKNN``. All support flexible inputs for the cell line view.

``SingleDrugKNN`` searches, for a test cell line, the most similar training cell lines and averages their
responses to the same drug. The distance is computed on the ``mrmr_k`` features (default 20) that are most relevant
for the response of this drug, selected by mRMR (:class:`drevalpy.models.utils.MRMRSelector`) on the training rows
only. ``mrmr_k: null`` uses all features of the gene list. In contrast to the pooled ``KNNRegressor``, drug
identity does not enter the distance.

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
   ``tcga_autoencoder_genes_available``, see below) and transformed to
   ``log2(x + 1)`` for the TPM datasets (CCLE, CTRPv1, CTRPv2; the others are already log scaled). In every
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
============================  ==========================================  ==================================

Both models fit the autoencoder inside each fold and only on training cell lines, so neither leaks
test expression into the encoder.

Autoencoder hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The four parameters DIPK also has keep DIPK's names, so that a configuration reads the same for
both models:

- ``epochs_autoencoder`` (default 2000), ``lr_autoencoder`` (1e-4), ``patience_autoencoder`` (50),
  ``batch_size_autoencoder`` (1024).

The cell lines of the early stopping split validate the autoencoder. The one parameter that only exists
here is ``autoencoder_cache_dir`` (unset): see below.

Gene list
^^^^^^^^^

**Motivation.** The default list ``tcga_autoencoder_genes_available`` is much larger than the 2,270 genes of
``gene_expression_intersection``. It was chosen for four reasons:

* The TCGA-pretrained encoder and the encoder trained from scratch on the cell lines must see the *same* feature
  space. With different gene sets, the effect of pretraining and the effect of the gene set would be confounded.
* The space is *deployment-optimal*: every gene is measurable by Ion Torrent RNA-seq (Ion AmpliSeq Transcriptome Human
  Gene Expression Panel), so new samples can be scored with the same model.
* Noise genes (not expressed or constant in the cell lines) are removed, but there is deliberately **no
  top-variance selection**: it would discard driver genes with a low variance. The reduction is left to the
  autoencoder bottleneck. Known landmark and clinical driver genes are protected from the noise filter.
* Every gene must exist in the expression data of all datasets (CTRPv2, GDSC1, PDX_Bruna, BeatAML2), so that all datasets share one space.

**Construction.** The list is a chain of set operations. Nothing in the chain uses the pretrained autoencoder or
response data, and only the last list is written, to ``<data_path>/meta/gene_lists/tcga_autoencoder_genes_available.csv``:

1. ``expression_intersection``: genes in the expression data of all datasets.
2. ``panel_intersection``: intersected with the genes of the Ion Torrent panel.
3. ``expressed_genes``: noise filter on the linear TPM expression of a TPM dataset (default CTRPv2),
   applied as ``log2(x + 1)``: a gene is dropped if its mean is below 0.5 or its variance below 1e-3, unless it is in
   the protect list (the 867 landmark genes of ``meta/gene_lists/landmark_genes.csv`` plus 26 clinical driver genes,
   893 genes; override with ``--protect <csv with column Symbol>``). The thresholds only make sense for linear TPM;
   the script raises an error if the noise dataset looks log scaled already.
4. ``tcga_autoencoder_genes_available``: minus the genes without a locus in the TCGA gene annotation
   (``gencode.v23.probemap.tsv``), which cannot be mapped to TCGA expression.

If the file is not in the metadata, the model creates it when it loads a dataset: from the expression of that dataset
alone, saved as ``tcga_autoencoder_genes_available_<dataset>.csv``. The noise filter then uses the dataset itself if it
is a TPM dataset, otherwise CTRPv2 if present, otherwise it is skipped. The lists of different datasets differ, so
models trained on one dataset and applied to another need the shared file.

**Inputs.** By default the script downloads the gene content of the panel: the public *AmpliSeq for Illumina
Transcriptome RNA Manifest* (dated 2019-03-13, anonymous, 20,802 genes; the ``[Targets]`` section, column ``Gene``)

.. code-block:: text

    https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/ampliseq-for-illumina/ampliseq-for-illumina-transcriptome-rna-manifest-20190313.zip

It has the same genes as the Ion AmpliSeq Transcriptome Human Gene Expression Panel (user guide
`MAN0010742 <https://assets.thermofisher.com/TFS-Assets/LSG/manuals/MAN0010742_AmpliSeqTranscriptomeHumanGeneExpr_UG.pdf>`_,
over 20,000 genes); the 10 ERCC spike-in controls of the Thermo Fisher BED file are not part of it and are not in the
data anyway, so the result is identical. Alternatively, pass the Thermo Fisher target file with ``--panel``
(``hg19_AmpliSeq_Transcriptome_21K_v1.bed``, ERCC variant; semicolon separated, no header, gene symbol in the seventh
column, adjustable with ``--panel_symbol_column``). The TCGA probemap (GENCODE v23, Xena Toil hub) is downloaded automatically; pass
``--probemap <gencode.v23.probemap.tsv>`` to use a local copy instead.

.. code-block:: bash

    python -m drevalpy.datasets.featurizer.create_autoencoder_gene_list --data_path data

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
