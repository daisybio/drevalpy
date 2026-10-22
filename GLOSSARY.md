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

**Dataset**:
A drug response screen together with the sample and drug features that belong
to it, stored as one file.

**View**:
One named feature table on one side, such as the sample view
`gene_expression` or the drug view `fingerprints`.
_Avoid_: omics view, feature view, omics layer, modality (except when describing
the stored file)

**Split mode**:
The rule that decides which pairs are held out for testing: leave-pair-out
(LPO), leave-sample-out (LSO), leave-tissue-out (LTO) or leave-drug-out (LDO).
_Avoid_: setting, test mode, leave-out mode, mode, LCO, leave-cell-line-out

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
The named output of a featurizer, handed to a predictor.

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

## Evaluation

**Experiment**:
One evaluation of several models over all folds of a split mode, optionally
with stress tests.
_Avoid_: benchmark (for this meaning)

**Run**:
One model trained and evaluated on one split.
_Avoid_: fold run, execution

**Trial**:
One hyperparameter configuration tried during tuning.
_Avoid_: sample, draw

**Stress test**:
An extra evaluation that perturbs the data to see how a model's score depends
on it: a randomization test or a robustness test.

**Randomization test**:
A stress test that permutes or replaces one view before training, to measure
how much the model relies on it.

**Robustness test**:
A stress test that repeats training on the same split, each repeat with its own
derived seed (so pair order, the model's own randomness and the early-stopping holdout
change), to measure how much the score varies between repeats.
_Avoid_: reseeding (the experiment's root seed does not change)
