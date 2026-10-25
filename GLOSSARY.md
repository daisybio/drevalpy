# DrEvalPy

DrEvalPy evaluates drug response prediction models: it trains models on sample
and drug data, scores them on held-out pairs, and compares them against
baselines under fixed split modes.

## Data

**Sample**:
One biological source of drug response measurements: a cell line, an organoid,
a patient-derived xenograft or primary patient cells.
_Avoid_: cell line (except for data that really are cell lines), biosample,
model system, cancer model

**Sample type**:
What kind of biological source a sample is: cell line, organoid, PDX or primary.

**Pair**:
One sample–drug combination with a measured response; the unit models train on
and are scored on.
_Avoid_: sample, observation, data point, instance

**Entity**:
A sample or a drug: what one row of a view describes.

**Side**:
Which kind of entity a view, a featurizer or a feature block is about: the
sample side or the drug side.

**Dataset**:
A drug response screen together with the sample and drug features that belong
to it, stored as one file.

**Study**:
One screen's dataset, such as GDSC2. In a dataset built with `concat`, the `study`
column of `obs` names the study each sample row came from.
_Avoid_: cohort, source

**View**:
One named feature table on one side, such as the sample view
`gene_expression` or the drug view `fingerprints`.
_Avoid_: omics view, feature view, omics layer, modality (except when describing
the stored file)

**Split mode**:
The rule that decides which pairs are held out for testing: leave-pair-out
(LPO), leave-sample-out (LSO), leave-tissue-out (LTO) or leave-drug-out (LDO).
_Avoid_: setting, test mode, leave-out mode, mode, LCO, leave-cell-line-out

**Fold**:
One of the splits a split mode produces: the pairs to train on, to stop early on
and to test on.

## Models

**Component**:
A featurizer or a predictor; the parts a model is assembled from.
_Avoid_: atom, building block, slot

**Featurizer**:
A component that turns one or more views of one side, sample or drug, into
feature blocks.

**Predictor**:
A component that turns feature blocks into drug response predictions.

**Feature block**:
The named output of a featurizer, handed to a predictor. Its rows are aligned
with the entities of its side, unless it is a global feature block. Its
annotations are free-form side information about the block, such as an ontology.

**Global feature block**:
A feature block that describes the whole side rather than one row per entity,
such as a gene–pathway map.
_Avoid_: metadata block

**Feature format**:
The shape of a feature block: a numeric matrix, a graph or a ragged sequence. A
featurizer declares the format it emits and a predictor the format it needs on
each side (`output_format`, `sample_input_format`, `drug_input_format`), so a
model whose parts disagree is rejected before training.
_Avoid_: contract, feature contract

**Prediction mode**:
Whether a model predicts a continuous response (regression) or a class
(classification).
_Avoid_: mode (alone), task

**Recipe**:
The short text form of a model configuration, naming its featurizers and its
predictor, such as `pca[gene_expression]:elasticNet`.
_Avoid_: recipe string, component stack, stack

**Model configuration**:
The full, unnamed description of which components a model uses and how they
are configured.
_Avoid_: architecture

**Model spec**:
A model configuration together with the name it is reported under. It holds no
trained state.
_Avoid_: model specification, model recipe

**Model**:
A model spec that can be trained, used for prediction, saved and loaded.

**Zoo preset**:
A model spec shipped with DrEvalPy or loaded from a zoo file, selected by its
zoo name.
_Avoid_: preset, zoo model, zoo entry, alias

**Zoo name**:
The name a zoo preset is selected and reported by.

**Naive baseline**:
A predictor that uses no features beyond the identities of samples and drugs,
used as the reference that other models must beat.
_Avoid_: baseline (alone), naive predictor, primary model

**Literature model**:
A model reimplemented from a published drug response method.
_Avoid_: published model, upstream model

**Response transformation**:
A reversible change to the drug response values a model trains on, undone on its
predictions, such as standard scaling or subtracting each drug's mean response.
_Avoid_: target transform, response transform

## Evaluation

**Experiment**:
One evaluation of several models over all folds of a split mode, optionally
with stress tests.
_Avoid_: benchmark (for this meaning)

**Run**:
One model trained and evaluated on one split.
_Avoid_: fold run, execution

**Metric**:
A score of predictions against observed responses, such as RMSE or the Pearson
correlation; one of a fixed list, each computed by a scipy or scikit-learn
function. Its **kind** (error, fit or correlation) decides when the score is
undefined and its **direction** whether lower or higher is better.
_Avoid_: mode (for the direction), measure, score function

**Grouping**:
How a run's pairs are split for scoring: by drug, by sample, by study, or by a
sample or drug annotation of the dataset such as tissue or drug class.

**Pooled score**:
A metric computed once over all pairs of a run.

**Per-group score**:
A metric computed separately over the pairs of each group of a grouping, then
averaged over the groups where it is defined. A group is undefined when its data
cannot support the metric (too few pairs, or a constant response); a constant
prediction scores 0 for a correlation.
_Avoid_: grouped metric, group score, stratified metric

**Trial**:
One hyperparameter configuration tried during tuning.
_Avoid_: sample, draw

**Stress test**:
An extra evaluation that perturbs the data to see how a model's score depends
on it: a randomization test or a robustness test.
_Avoid_: diagnostic

**Randomization test**:
A stress test that permutes or replaces one view before training, to measure
how much the model relies on it.

**Robustness test**:
A stress test that repeats training on the same split, each repeat with its own
derived seed (so pair order, the model's own randomness and the early-stopping holdout
change), to measure how much the score varies between repeats.
_Avoid_: reseeding (the experiment's root seed does not change)

**Robustness repeat**:
One of the runs of a robustness test, numbered from 0.
_Avoid_: robustness trial, trial (a trial is one hyperparameter configuration)
