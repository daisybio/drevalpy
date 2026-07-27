Datasets and response data
==========================

DrEvalPy ships commonly used drug-response screens plus small toy subsets for
fast local checks. Each built-in name resolves to a response table and the
matching cell-line and drug feature views used by the model zoo.

Built-in datasets
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 18 14 22 28

   * - Dataset
     - Curves
     - Drugs
     - Samples
     - Notes
   * - GDSC1
     - 316506
     - 378
     - 970 cell lines
     - Genomics of Drug Sensitivity in Cancer v1
   * - GDSC2
     - 234437
     - 287
     - 969 cell lines
     - Genomics of Drug Sensitivity in Cancer v2
   * - CCLE
     - 11670
     - 24
     - 503 cell lines
     - Cancer Cell Line Encyclopedia
   * - CTRPv1
     - 60758
     - 354
     - 243 cell lines
     - Cancer Therapeutics Response Portal v1
   * - CTRPv2
     - 395025
     - 546
     - 886 cell lines
     - Cancer Therapeutics Response Portal v2
   * - TOYv1
     - 2711
     - 36
     - 90 cell lines
     - Small CTRPv2 subset for testing
   * - TOYv2
     - 2784
     - 36
     - 90 cell lines
     - Second toy set for cross-study checks
   * - BeatAML2
     - 62487
     - 166
     - 569 patients
     - Ex vivo AML patient screens
   * - PDX_Bruna
     - 2559
     - 104
     - 37 mouse passages
     - Breast cancer PDTX-derived cultures

Response measures
-----------------

Built-in screens expose the usual dose–response summaries: ``LN_IC50``,
``EC50``, ``IC50``, ``pEC50``, ``AUC``, and ``response``.

When CurveCurator refitting is enabled (the default for built-in data), the
target column name gains a ``_curvecurator`` suffix (for example
``LN_IC50_curvecurator``). Those refit measures use one shared fitting
procedure across studies and are preferred for cross-study comparability.
Original published values remain available when you opt out of refitting.

Custom data concepts
--------------------

You can also evaluate against custom response tables. Conceptually there are
two shapes:

* **Raw viability (long format).** Columns ``dose``, ``response``, ``sample``,
  and ``drug``, plus an optional ``replicate``. Dosages must be in µM.
  DrEvalPy fits curves with the same CurveCurator procedure used for built-in
  refits and then exposes the usual ``*_curvecurator`` measures.
* **Prefit response CSV.** At least ``cell_line_id``, ``drug_id``, and a
  measure column. Leave-Tissue-Out also needs a ``tissue`` column.

Custom names are not restricted to the built-in list; the library treats an
unknown dataset name as a custom load path.

Feature provenance
------------------

Cell-line and drug features shipped with the built-in screens come from public
omics and chemistry sources (expression, methylation, mutation, CNV,
proteomics, fingerprints, and model-specific embeddings). Preprocessing and
feature inventories live in the
`preprocess_drp_data <https://github.com/daisybio/preprocess_drp_data>`_
repository.

How to load and run
-------------------

- :doc:`/cli/experiment` — CLI experiment options for dataset and measure
- :doc:`/python/datasets` — Python dataset loading and custom tables
