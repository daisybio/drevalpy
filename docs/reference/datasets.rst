Datasets and file format
========================

A dataset is one ``.h5mu`` (MuData) file holding the drug response and the features of the
samples and drugs. ``drevalpy.load`` returns it as a ``Dataset``.

Built-in datasets
-----------------

``drevalpy list datasets`` prints the registered datasets with the columns Name, Source, File
and Origin. These rows are built in. They all come from the source ``orakl``, the base URL
``s3://orakl-open-source-data/drevalpy_h5mu/gen3``, and their file is ``<Name>.h5mu``.

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
     - The drugs, indexed by PubChem ID, with the columns ``drug_name`` and ``canonical_smiles``
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

Curve-quality filter
--------------------

Every splitter keeps only the pairs that pass a fixed curve-quality filter:

.. code-block:: text

   relevance_score >= -log10(0.05)   (1.3010299956639813)
   abs(fold_change) >= 0.45

A pair whose ``relevance_score`` or ``fold_change`` is NaN fails. The thresholds cannot be
changed per split. ``drevalpy.data.curve_quality_mask`` computes the same mask and takes
optional thresholds for the other quality layers; passing ``None`` for a threshold switches
that check off.

Synthetic dataset
-----------------

``drevalpy.synthetic(*, name="SYNTHETIC", n_samples=60, n_drugs=8, n_tissues=6, omics=None,
feature_names=None, n_features=16, missing_fraction=0.05, low_quality_fraction=0.0, seed=3)``
returns an in-memory ``Dataset``. It has at most 31 drugs and six tissues. Its responses
make drugs differ strongly in mean potency, so a pooled score is high even for a model that
ranks samples poorly within a drug. ``omics`` names the sample modalities to add, as a list or
as a mapping from name to the number of covered samples.

``drevalpy data synthetic --output FILE`` writes the default dataset: 60 samples named
``SYNTH-000`` onward, 8 drugs with IDs from ``100000``, 456 measured pairs and only a
16-feature ``gene_expression`` modality. Its ``obs`` has only the ``tissue`` column.
