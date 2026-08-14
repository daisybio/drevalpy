Datasets and response data
==========================

DrEvalPy ships commonly used drug-response screens. Each built-in name resolves
to a response table and the matching cell-line and drug feature views used by
the model zoo.

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

``response.X`` holds ``pEC50``. Every other dose–response summary the fit
produces is a named ``response.layers`` entry: ``LN_IC50``, ``EC50``, ``IC50``,
``AUC``, plus the quality metrics below. Published, non-refit values from the
original study are carried alongside with a ``_published`` suffix where the
source provides them.

All built-in screens are refit with one shared CurveCurator procedure, which is
what makes numbers comparable across studies.

Custom data concepts
--------------------

You can also evaluate against custom response tables. Conceptually there are
two shapes:

* **Raw viability (long format).** Columns ``dose``, ``response``, ``sample``,
  and ``drug``, plus an optional ``replicate``. Dosages must be in µM.
  DrEvalPy fits curves with the same CurveCurator procedure used for built-in
  refits, so the same measures become available.
* **Prefit response CSV.** At least ``cell_line_id``, ``drug_id``, and a
  measure column. Leave-Tissue-Out also needs a ``tissue`` column.

Custom names are not restricted to the built-in list; the library treats an
unknown dataset name as a custom load path.

On-disk format: MuData
-----------------------

Datasets are stored as `MuData <https://mudata.readthedocs.io/>`_ objects
(``.h5mu`` files). MuData is a multimodal extension of
`AnnData <https://anndata.readthedocs.io/>`_ — the standard data structure in
the single-cell genomics ecosystem. While DrEvalPy works with bulk data (cell
lines, not single cells), the underlying data layout is identical: an
observation-by-variable matrix with metadata on both axes.

A single MuData object bundles all modalities of a dataset:

.. code-block:: text

   MuData
   ├── response        # AnnData: cell_lines × drugs (X = IC50/AUC/...)
   ├── expression      # AnnData: cell_lines × genes
   ├── methylation     # AnnData: cell_lines × CpG regions
   ├── mutations       # AnnData: cell_lines × genes
   ├── copy_number     # AnnData: cell_lines × genes
   ├── proteomics      # AnnData: cell_lines × proteins
   └── fingerprints    # AnnData: drugs × fingerprint bits

Each modality is an AnnData with:

- ``X`` — the data matrix (e.g. expression counts, response values)
- ``obs`` — observation (cell line / drug) metadata (tissue, name, ...)
- ``var`` — variable (gene / drug) metadata
- ``layers`` — alternative representations (e.g. raw vs normalized counts)

.. _curve-quality:

Curve quality metrics
---------------------

The built-in screens are refit with CurveCurator, and the ``response`` modality
ships **every** fitted curve — including the ones the fit itself says are
meaningless. Because ``X`` holds ``pEC50``, which exists for every curve that
converged, "not NaN" does not mean "trustworthy": the quality metrics stored
alongside as ``response.layers`` are what separate the two.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Layer
     - Better
     - Meaning
   * - ``relevance_score``
     - higher
     - SAM-corrected significance. This is the multiple-testing-corrected
       statistic, so prefer it over ``p_value``.
   * - ``fold_change``
     - larger ``abs``
     - Curve fold change, already log2. The magnitude is the effect size.
   * - ``p_value``
     - lower
     - Raw F-test p-value, **uncorrected**.
   * - ``log_p_value``
     - higher
     - ``-log10(p_value)``, also uncorrected.
   * - ``f_value`` / ``f_value_sam``
     - higher
     - F statistic of the fit, and its s0-corrected counterpart.
   * - ``R2`` / ``RMSE``
     - higher / lower
     - Goodness of fit.
   * - ``signal_quality``
     - higher
     - Signal quality of the underlying measurements.
   * - ``slope``, ``front``, ``back``
     - see note
     - Curve shape. A slope pinned at the fitting bound describes a step, which
       is usually an artefact; ``front`` and ``back`` are the fitted plateaus.
   * - ``regulation``
     - —
     - CurveCurator's own verdict, encoded ``up = 1``, ``down = -1``,
       ``not = 0``, and NaN where it reached none.
   * - ``pec50_error``, ``slope_error``, ``front_error``, ``back_error``
     - lower
     - Standard error of each fitted parameter, from a Moore-Penrose
       pseudo-inverse of the fit's Jacobian. ``pec50_error`` is the uncertainty
       on ``X`` itself, and these are the only per-curve uncertainty estimates
       the pipeline produces.

Every built-in splitter drops the pairs that fail

.. code-block:: text

   relevance_score >= -log10(0.05)   and   abs(fold_change) >= 0.45

which are the ``alpha`` and ``fc_lim`` that DrEvalPy passes to CurveCurator, so
the rule reproduces ``regulation != 0`` exactly. This is not configurable per
split — a comparison between models is only meaningful over the same pairs.

Filtering on any other metric, or on other thresholds, is available through
:func:`~drevalpy.plugin.curve_quality_mask`; see :doc:`/python/extensions`.

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

- :doc:`/cli/datasets` — CLI dataset loading and splitting
- :doc:`/python/datasets` — loading and splitting in Python
- :doc:`/python/quickstart` — end-to-end Python track map after concepts
