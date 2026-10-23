Datasets and file format
========================

A dataset is one ``.h5mu`` (MuData) file holding the drug response and the features of the
samples and drugs. ``drevalpy.load`` returns it as a ``Dataset``.

Built-in datasets
-----------------

``drevalpy list datasets`` prints the registered datasets with the columns Name, Source, File
and Origin. These rows are built in. The screens come from the source ``orakl``, the base URL
``s3://orakl-open-source-data/drevalpy_h5mu/gen3``, and ``TOYv1`` from ``orakl_gen4``, the base URL
``s3://orakl-open-source-data/drevalpy_h5mu/gen4``. Their file is ``<Name>.h5mu``.

.. list-table::
   :header-rows: 1
   :widths: 16 30 14 14 26

   * - Name
     - Screen
     - Pairs
     - Drugs
     - Samples
   * - ``GDSC1``
     - Genomics of Drug Sensitivity in Cancer v1
     - 316129
     - 378
     - 969 cell lines
   * - ``GDSC2``
     - Genomics of Drug Sensitivity in Cancer v2
     - 233002
     - 285
     - 969 cell lines
   * - ``CTRPv1``
     - Cancer Therapeutics Response Portal v1
     - 60757
     - 354
     - 242 cell lines
   * - ``CTRPv2``
     - Cancer Therapeutics Response Portal v2
     - 386372
     - 545
     - 885 cell lines
   * - ``BeatAML2``
     - Ex vivo AML patient screens
     - 57727
     - 166
     - 569 patients
   * - ``PDX_Bruna``
     - Breast cancer PDTX-derived cultures
     - 2549
     - 104
     - 37 mouse passages

``TOYv1`` is also registered: a subsample of 32 cell lines and 12 drugs (384 pairs) for quick
checks.

Pairs counts the sample-drug pairs with a pEC50 value.

A dataset registered with ``register_dataset`` or a declaration file has the origin
``custom``; see :doc:`extension_files`.

Loading
-------

``drevalpy.load(name)`` tries these in order and raises ``FileNotFoundError`` if none works:

#. The file ``<cache dir>/<name>.h5mu``.
#. A registered name: its cached file, else a download into the cache directory.
#. A path to an existing ``.h5mu`` file.

A registry entry may carry a ``sha256``, as ``TOYv1`` does. A download that does not match it
raises an error and is deleted, and a cached file that does not match it is deleted and
downloaded again; a cached file that matches is used without network access.

The cache directory is set by ``DREVALPY_CACHE_DIR``; see :doc:`environment`. A cached file is
deleted and downloaded again if it cannot be parsed, or if it lacks the curve-quality layers
listed below. A file given by path is used as it is. DrEvalPy has no loader for CSV or other
formats; ``drevalpy curate`` fits dose-response curves and writes an ``.h5ad`` file instead.

``drevalpy data load NAME OUTPUT`` writes a loaded dataset to an ``.h5mu`` file.
``drevalpy.synthetic`` and ``drevalpy data synthetic`` build a small simulated dataset that
needs no download.

File layout
-----------

``response`` modality
~~~~~~~~~~~~~~~~~~~~~

The file must have a modality named ``response``; without it loading raises ``KeyError``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Part
     - Content
   * - ``obs``
     - The samples
   * - ``var``
     - The drugs, indexed by PubChem ID, with the columns ``drug_name`` and ``canonical_smiles``.
       The curated datasets give each drug ID one canonical SMILES string and each InChIKey one
       drug ID (the lowest PubChem ID), so datasets of different studies agree on a drug
   * - ``X``
     - The pEC50 of each sample and drug
   * - ``layers``
     - The other dose-response summaries and quality metrics, listed below
   * - ``varm``
     - Drug features. A key may be written ``name:variant``
   * - ``obsm["pathway_features"]``
     - Pathway features of the samples

Layers written by ``drevalpy curate``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Group
     - Layers
   * - Response summaries
     - ``EC50``, ``IC50``, ``LN_IC50``, ``AUC``
   * - Curve quality
     - ``relevance_score``, ``fold_change``, ``p_value``, ``log_p_value``, ``f_value``,
       ``f_value_sam``, ``R2``, ``RMSE``, ``signal_quality``, ``slope``, ``front``, ``back``,
       ``regulation``
   * - Parameter errors
     - ``pec50_error``, ``slope_error``, ``front_error``, ``back_error``

Sample feature modalities
~~~~~~~~~~~~~~~~~~~~~~~~~

Each further modality holds one kind of sample features, with its ``obs`` aligned to the
samples. The public names are:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Public name
     - Modality key in the file
   * - ``gene_expression``
     - ``gene_expression``
   * - ``methylation``
     - ``methylation``
   * - ``mutations``
     - ``mutations``
   * - ``proteomics``
     - ``proteomics``
   * - ``copy_number_variation_gistic``
     - ``copy_number_variation``

A file may store the modality under the public name or under the key in the second column;
DrEvalPy uses whichever the file has. Recipes and configs use the public names; see
:doc:`recipes`.

Gene lists
~~~~~~~~~~

A gene list is a CSV of gene symbols that ships inside DrEvalPy, in
``drevalpy/components/featurizers/sample/gene_lists/``. It is the only source: a dataset never
carries gene lists, so a list name selects the same genes on every dataset. A featurizer that
takes a ``gene_list`` keeps exactly the genes of that list, in the order of the list, so two
datasets with different gene columns give the same feature order.

The featurizers ``scaledGeneExpression``, ``pharmaFormerGeneExpression`` and ``dipkGeneExpression``
take ``gene_list: <name>``; ``None``, the default, keeps the whole view. ``molirOmics`` and
``superfeltrOmics`` take ``gene_lists`` mapping a view name to a list name. ``landmarkGenes`` takes
``gene_list_stem`` (default ``landmark_genes``) and ``landmarkGenesReduced`` always uses
``landmark_genes_reduced``.

A list that names genes the view lacks is an error, not a smaller selection::

   Error: view 'gene_expression' lacks 812 of 867 genes in 'landmark_genes' (missing: ...).
   Pass gene_list=None to use every gene of the view.

The shipped lists, by the view they apply to:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - List
     - Genes and purpose
   * - ``landmark_genes``
     - 867 L1000 landmark genes; the expression input of the sklearn models, SimpleNN,
       DrugGNN, PharmaFormer, EnsembleMF and SingleDrugKNN in 1.6.1
   * - ``landmark_genes_reduced``
     - The landmark genes also measured in the copy-number, mutation and proteomics
       intersections
   * - ``gene_expression_intersection``
     - Genes in the expression view of every cell-line screen (CTRPv1, CTRPv2, GDSC1, GDSC2); the
       input of DIPK
   * - ``gene_list_paccmann_network_prop``
     - 1,955 genes of the PaccMann network-propagation selection that the cell-line screens measure
   * - ``gene_list_paccmann_network_prop_reduced``
     - The PaccMann list restricted to the reduced gene universe
   * - ``tcga_autoencoder_genes_available``
     - 10,946 genes measured in CTRPv2, GDSC1, PDX_Bruna and BeatAML2, covered by the Ion AmpliSeq
       panel, expressed above noise level and present in the TCGA gene annotation; the input of
       the TCGA-pretrained autoencoder
   * - ``drug_target_genes_all_drugs``, ``drug_target_genes_reduced``
     - Drug-target genes; the mutation and copy-number input of the multi-view models
   * - ``mutations_intersection``, ``copy_number_variation_gistic_intersection``,
       ``methylation_intersection``, ``proteomics_intersection``
     - Features shared by every dataset that has the view; the MOLIR inputs
   * - ``landmark_genes_proteomics``, ``drug_target_genes_all_drugs_proteomics``,
       ``gene_list_paccmann_network_prop_proteomics``
     - The curated lists restricted to the proteomics intersection

``drevalpy.components.featurizers.sample.gene_lists.resolve_gene_list(name)`` returns the symbols
of a list and raises ``UnknownNameError`` for a name that is not shipped, listing those that are.
``select_genes(source, view, name)`` applies a list to a feature source and raises
``InvalidInputError`` for the missing genes. ``GENE_LISTS_BY_VIEW`` assigns each list to its view.
``_make_gene_lists.py`` and ``_make_autoencoder_gene_list.py`` in the same directory regenerate the
CSVs; the second needs the network and the dataset files.

The lists are those of drevalpy 1.6, minus the genes the re-curated cell-line screens no longer
measure: ``PRRC2B`` from ``gene_expression_intersection``, ``PRRC2B`` and ``S100A7L2`` from
``gene_list_paccmann_network_prop`` and 40 genes from
``copy_number_variation_gistic_intersection``. PDX_Bruna and BeatAML2 lack further genes of
several lists, including ``landmark_genes``, so presets that use those lists raise the
missing-genes error on them; pass ``gene_list=None`` to use their whole expression view.

Global annotations
~~~~~~~~~~~~~~~~~~

``mdata.obs`` has a unique sample-ID index. The name of the index is not read.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``tissue``
     - Required. The tissue of each sample; the ``LTO`` split mode and the tissue-based
       predictors read it
   * - ``cellosaurus_id``, ``cell_line_name``
     - Optional. Present only in datasets whose samples are cell lines
   * - ``study``, ``source_sample_id``
     - Written by ``concat``; see `Concatenated datasets`_

``mdata.uns`` keys DrEvalPy reads or writes:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Key
     - Content
   * - ``dataset_name``
     - The dataset name
   * - ``sample_featurizer_variants``, ``drug_featurizer_variants``
     - Featurizer outputs stored by ``Dataset.precompute()``
   * - ``drug_graphs``
     - Molecular graphs of the drugs
   * - ``pathways_gmt``
     - Pathway definitions
   * - ``randomization``, ``randomization_provenance``
     - Written on randomized copies; see :doc:`split_modes`

Concatenated datasets
---------------------

``drevalpy.data.concat(datasets, *, join="exact", name=None)`` and
``drevalpy data concat DATASETS... OUTPUT [--join exact|inner] [--name NAME]`` join the
datasets of several studies into one ordinary dataset; ``drevalpy run`` does the same when it is
given several ``--dataset`` options. How to use it is in :doc:`/how-to/run_cross_study`.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Part
     - Result
   * - Study
     - The name of each input dataset. The names must be distinct, non-empty and free of ``::``
   * - Sample IDs
     - ``<study>::<sample_id>``. A sample measured in two studies is two rows, one per study
   * - ``obs``
     - The inputs' columns, plus ``study`` (categorical) and ``source_sample_id``, the ID in the
       input. An input that already has either column is rejected
   * - Sample views
     - Stacked by rows, view by view
   * - Drugs
     - The union of the drug IDs. A drug ID with different ``canonical_smiles`` in two inputs
       raises ``ValueError``
   * - Response and layers
     - The union of the layers, with ``NaN`` for a pair the study did not measure
   * - ``uns["dataset_name"]``
     - ``name``, or the input names joined by ``+``, such as ``GDSC2+CTRPv2``
   * - Not carried over
     - Precomputed featurizer variants and the ``obsm`` and ``varm`` embeddings; ``concat`` warns

``join="exact"`` requires every view that the inputs share to have identical feature names, in
the same order; ``join="inner"`` keeps the features common to all inputs. With ``"exact"`` a view
that only some inputs have raises; with ``"inner"`` it is dropped with a warning. Different units
of a view always raise. Randomized datasets cannot be concatenated, and the inputs are not
changed.

Curve-quality filter
--------------------

``split`` and ``run`` pass a curve-quality rule to the splitter, and the built-in split modes
use only the pairs that pass it. The rule is a ``drevalpy.data.CurveQuality``. The default is the CurveCurator rule:

.. code-block:: text

   relevance_score >= -log10(0.05)   (1.3010299956639813)
   abs(fold_change) >= 0.45

A pair whose ``relevance_score`` or ``fold_change`` is NaN fails. Pass the rule as the
``quality`` argument of ``drevalpy.split`` and ``drevalpy.run``; ``None`` means
``CurveQuality.default()``. The rule is recorded in the metadata of each fold under ``quality``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Rule
     - How to get it
   * - Default
     - ``CurveQuality.default()``; on the command line, no option
   * - Other thresholds
     - ``CurveQuality(min_relevance_score=2.0, min_r2=0.8)`` or
       ``dataclasses.replace(CurveQuality.default(), ...)``; on the command line,
       ``--min-relevance-score`` and ``--min-abs-fold-change``
   * - No rule
     - ``CurveQuality.off()``; on the command line, ``--no-quality-filter``. Every pair with a
       finite response is kept and no quality layer is read

``CurveQuality`` is a frozen dataclass with 15 numeric thresholds, each ``None`` when it is not
checked, and ``regulation``, the labels (``"up"``, ``"down"``, ``"not"``) to keep. The thresholds
beyond the default two are ``max_p_value``, ``min_log_p_value``, ``min_f_value``,
``min_f_value_sam``, ``min_r2``, ``max_rmse``, ``min_signal_quality``, ``min_abs_slope``,
``max_abs_slope``, ``min_front``, ``max_back``, ``min_pec50`` and ``max_pec50``. ``mask(dataset)``
returns the boolean pair mask.

A rule that needs a response layer the dataset lacks raises ``UnknownNameError`` (a ``KeyError``). A dataset without the
quality layers, such as one with only a pEC50 matrix, therefore needs ``CurveQuality.off()``
(``--no-quality-filter``). ``run`` and ``data split`` accept the three options above; ``single``
reads the folds it is given and has none. ``--no-quality-filter`` cannot be combined with
``--min-relevance-score`` or ``--min-abs-fold-change``.

``drevalpy.data.curve_quality_mask`` remains and computes the same mask as ``CurveQuality`` with
optional thresholds. ``drevalpy.plugin.observed_mask(dataset, quality)`` is the mask a splitter
reads: the pairs with a finite response that also pass ``quality``.

Synthetic dataset
-----------------

``drevalpy.synthetic(*, name="SYNTHETIC", n_samples=60, n_drugs=8, n_tissues=6,
omics=("gene_expression",), sample_views=(), drug_views=(), uns_payloads=(), extra_layers=(),
feature_names=None, n_features=16, missing_fraction=0.05, low_quality_fraction=0.0, seed=3)``
returns an in-memory ``Dataset``. It has at most 31 drugs and six tissues. Its responses
make drugs differ strongly in mean potency, so a pooled score is high even for a model that
ranks samples poorly within a drug. It carries the curve-quality layers, and
``low_quality_fraction`` makes that share of the pairs fail the default rule. Give each dataset a
different ``name`` to concatenate several of them; datasets with the same ``seed`` describe the
same underlying screen, and different seeds permute the drug effects.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Argument
     - Meaning
   * - ``omics``
     - The sample modalities to add, from ``gene_expression``, ``proteomics``, ``mutations``,
       ``methylation`` and ``copy_number_variation_gistic``. A list, or a mapping from name to the
       number of covered samples. ``None`` or an empty sequence adds none, leaving the response
       only
   * - ``feature_names``, ``n_features``
     - Column names. By default each view holds every gene of the gene lists shipped for it, so any
       ``gene_list`` finds its genes: about 11,600 for ``gene_expression``. ``n_features`` is a
       minimum and only adds placeholder names beyond that; it also sets how many genes the
       ``pathways_gmt`` and ``sparsego`` payloads cover
   * - ``sample_views``
     - Views in ``response.obsm``: ``pathway_features``
   * - ``drug_views``
     - Views in ``response.varm``: ``morgan_fingerprint`` (needs ``rdkit``), ``chemberta``,
       ``bpe_smiles``, ``smilesvec``
   * - ``uns_payloads``
     - Entries of ``mdata.uns``: ``drug_graphs`` (needs ``rdkit``), ``pathways_gmt``,
       ``sparsego``, ``bpe_codes`` (needs ``subword-nmt``)
   * - ``extra_layers``
     - Response layers derived from the pEC50 matrix, from ``AUC`` and ``IC50``

``drevalpy data synthetic --output FILE`` writes the default dataset: 60 samples named
``SYNTH-000`` onward, 8 drugs with IDs from ``100000``, 456 measured pairs and only a
``gene_expression`` modality with every gene of the shipped expression gene lists. Its ``obs`` has only the ``tissue`` column. The command
has the options ``--seed``, ``--n-samples``, ``--n-drugs``, ``--n-tissues`` and ``--name``
(default ``SYNTHETIC``); give two files different names to concatenate them.
