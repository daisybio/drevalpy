Where the datasets come from
============================

A drug response model is only as comparable as its data. This page explains what
the built-in datasets are, how their responses are defined, and why DrEvalPy
refits dose-response curves instead of using each study's published numbers. The
lists of datasets with their sizes and file names are in :doc:`/reference/datasets`,
and loading your own data is covered in :doc:`/how-to/bring_your_own_data`.

The built-in datasets
---------------------

Seven names are registered under the built-in source:

- **GDSC1** and **GDSC2**: the Genomics of Drug Sensitivity in Cancer screens,
  two successive versions of the screen. They are in-vitro screens of established
  cancer samples.
- **CTRPv1** and **CTRPv2**: two versions of the Cancer Therapeutics Response
  Portal, also in-vitro screens of established cancer samples.
- **BeatAML2**: ex vivo screens of primary patient samples from acute myeloid
  leukaemia. The samples are patient material, not established lines.
- **PDX_Bruna**: a screen of patient-derived xenograft models of breast cancer.
  The samples are patient-derived xenografts, not established lines.
- **TOYv1**: a toy dataset for trying out the tools.

"Sample" is the general word throughout DrEvalPy because the biological source
differs between these datasets. A sample is an established line in GDSC and CTRP, a
patient's cells in BeatAML2 and a xenograft in PDX_Bruna. Models trained on one kind
of sample cannot be assumed to transfer to another, which is a reason to look at
the leave-sample-out and leave-tissue-out split modes (see
:doc:`evaluation_design`).

The datasets are not bundled with the package. They are downloaded on first use
from the built-in source and cached locally; see :doc:`registries_and_plugins`
for how sources work and what access they need.

Datasets are MuData files
-------------------------

Each dataset is a single MuData file (``.h5mu``). MuData bundles several tables
that share their sample or drug axis, so that one file holds everything a model
might read:

- a response table with samples as rows and drugs as columns;
- one table of features per sample view, such as ``gene_expression``;
- one table of features per drug view, such as molecular fingerprints.

Because the views live in one file, featurizers can ask for a view by name, and a
split mode can find the tissue labels it needs in the same place. The views a
dataset provides are not the same for every dataset. Gene expression is present in
all six screens. Copy number is present in the CTRP and GDSC screens and in
PDX_Bruna, while methylation and mutations are available in the CTRP and GDSC
screens only. A model whose featurizer needs a missing view cannot be run on that
dataset.

How a response is defined
-------------------------

A drug screen measures cell viability at several concentrations of a drug. A
*response* is a summary of the resulting dose-response curve, and different
summaries disagree. Published values are also computed with different curve
fitting procedures in different studies, so two screens can disagree on the same
drug and sample for reasons that have nothing to do with biology.

DrEvalPy therefore refits the raw curves of every built-in screen with one shared
procedure, CurveCurator, so the response definition is the same in all of them.
The primary response in the response table is the pEC50, the negative logarithm of
the concentration at half-maximal effect. The other summaries from the same fit,
such as the LN_IC50, the EC50, the IC50 and the area under the curve, are kept as
named layers next to it. Where a study published its own values, they are carried
along in layers with a ``_published`` suffix, so you can compare, but they are not
the ones models are trained on by default.

Fitting a curve always produces a number, including when the curve is flat or the
fit is meaningless. A pEC50 that is not missing is therefore not the same as a
pEC50 that can be trusted. Each fitted curve carries quality metrics that measure
the significance of the fit, the effect size, the goodness of fit and the
uncertainty of the fitted parameters.

Why pairs are filtered on curve quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the built-in split modes drop pairs whose curve fails two tests: the
multiple-testing corrected relevance score must be at least ``-log10(0.05)``, and the absolute fold change must be at least 0.45. These are the thresholds that
DrEvalPy passes to CurveCurator, so the rule agrees with CurveCurator's own
verdict that a curve is regulated. Drugs that do nothing to a sample leave noise
behind, and models are scored on pEC50 values only where the curve shows an
effect.

The rule is one setting of an experiment, not of a model or a splitter. A comparison
between models is only meaningful over the same pairs, so every model of a run is
trained and scored on the pairs that pass the same rule, and the rule is recorded in
the metadata of each fold. You can change it for the whole run: other thresholds, any
of the further quality metrics, or no rule at all for a dataset that carries only a
pEC50 matrix. See :doc:`/reference/datasets`.

Bringing your own responses
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your own screen can enter in two ways. If you have raw viability measurements, a
curation step fits curves with the same CurveCurator procedure and produces the
same response measures and quality metrics; the input needs the drug, the sample,
the concentration and the measured intensity, with an optional replicate. If you
already have fitted responses, you provide a table of sample, drug and the measure
directly. Leave-tissue-out splits also need a tissue for each sample. A dataset
name that is not in the built-in list is treated as a custom load path. The
step-by-step instructions are in :doc:`/how-to/bring_your_own_data`.

Features and their origin
-------------------------

The sample and drug features shipped with the built-in datasets come from public
sources: omics data for the samples and chemical descriptors for the drugs. Their
preprocessing and the inventory of features are maintained in a separate
repository, `preprocess_drp_data <https://github.com/daisybio/preprocess_drp_data>`_.
A few featurizers use additional external artifacts, such as protein-interaction
tables or pretrained molecular embeddings. These are downloaded on first use into
the local cache and are located through ``DREVALPY_ARTIFACTS_URI`` (see
:doc:`/reference/environment`).
