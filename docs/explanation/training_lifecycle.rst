How a model trains: scopes and early stopping
=============================================

Every model in DrEvalPy goes through the same training procedure, whether it is
a naive baseline, an ElasticNet or a neural literature model. This page explains
what data each stage of that procedure may touch and why. Other pages refer here
for the three *scopes* (called A, E and V below) and for the three pipeline
states.

Three scopes carved out of one fold
-----------------------------------

A split mode produces, for each fold, a training set, a validation set and a
test set of pairs (see :doc:`evaluation_design`). The test set is never used to
make a decision about a model. The validation set is cut again so that it can
serve two jobs without one contaminating the other:

**A (fit)**
   The training pairs. A model's weights, and everything fitted on data, are
   learned from A.

**E (early stopping)**
   The first 25% of the validation pairs, with at least one pair. Training loops
   that stop when a score stops improving watch E.

**V (validation)**
   The rest of the validation pairs. During hyperparameter tuning, V is the set
   a trial is scored on to choose between configurations.

A model can only be early-stopped on data it was not fit on, and a tuned
configuration should not be chosen on the same data the stopping rule has already
shaped. Splitting validation into E and V keeps those two uses apart.

.. mermaid::

   flowchart LR
       fold[one fold] --> train[train pairs]
       fold --> val[validation pairs]
       fold --> test[test pairs]
       val --> E[E: first 25 percent]
       val --> V[V: remaining pairs]
       train --> A[A: fit]

Where each scope is used
~~~~~~~~~~~~~~~~~~~~~~~~

Hyperparameter tuning and the final fit use the scopes differently:

- *Each tuning trial* trains on A, may stop early on E, and is scored on V. Trials
  run one after another, and the sampler is seeded (see :doc:`reproducibility`).
- *The final fit* uses the winning hyperparameters. It trains on A together with
  V, since V is no longer needed to choose anything, and it still stops early on
  E.
- *Evaluation* uses the test pairs, which are disjoint from A, E and V.

E and V only need to be disjoint as sets of pairs. They may share samples or
drugs, because both come from the same validation split. The split-mode checks
(for example, that a leave-sample-out test set shares no sample with training) run
on the original masks before E and V are derived.

The scopes, their counts and checksums are recorded with every fold's metadata,
so a result file records which pairs were used for fitting, stopping, validation
and testing.

Early stopping is a model property
----------------------------------

Not every model has a training loop that can stop early. Each model configuration
declares whether it supports early stopping. When it does and the validation set
has more than one pair, E and V are derived as above. Otherwise the whole
validation set is V and nothing is held out for stopping.

Some models go further and *require* an early-stopping set. DIPK and
PharmaFormer, and the featurizer that DIPK ships with, select a checkpoint from
E; if E ends up empty after pairs with missing inputs are removed, the fit raises
an error rather than silently training without it. SRMF, Precily and SparseGO do
not support early stopping: their literature versions train for a fixed budget
or use their own solver.

How the literature models use E to pick checkpoints, and where that differs from
the original papers, is described in :doc:`literature_models`.

Nothing is fitted on data outside A
-----------------------------------

The same discipline applies to the preprocessing that sits around a model:

- *Featurizers* are fitted on the training entities only. A scaler, a principal
  component analysis or a vocabulary is learned from training samples and drugs
  and then applied unchanged to validation and test entities. Because the
  featurizers are fitted per fold, per trial and per final fit, the model
  configuration is reusable and holds no fitted state itself (see
  :doc:`models_from_components`).
- *Pairs a featurizer rejects* are dropped before fitting. Their number can
  differ between folds.
- *The response transform* (none, standard, minmax, robust, drug_mean or
  drug_tissue_mean) is fitted on the training targets only, using a fresh copy for
  each fit. The two group means also take the drug of each pair, and the tissue of
  its sample, from the dataset being fitted or predicted. The predictor learns in
  transformed units and its predictions are converted back to the original units
  before scoring, so metrics are always reported in the units of the data.

Pipeline states
---------------

A trained model's internals live in a private pipeline object with three states.

``UNTRAINED``
   Nothing has been fitted. Calling ``predict`` raises an error.

``EMPTY``
   Training ran but there were no usable pairs, for example because every pair
   was rejected by a featurizer. The model is not broken, but it has nothing to
   offer: ``predict`` returns a missing value (NaN) for every requested pair.
   This keeps a fold with no usable data from crashing a whole experiment while
   still showing up in the results as missing predictions.

``FITTED``
   The model has learned from at least one pair and can predict. Only a fitted
   model can be saved.

Every call to ``train`` starts from scratch. It resets the pipeline to
``UNTRAINED`` and builds fresh featurizers and a fresh predictor, so one model
object used for several folds or trials never carries state from one fit to the
next. If training fails midway, the pipeline stays ``UNTRAINED``.

Saving and loading
~~~~~~~~~~~~~~~~~~

Saving a fitted model writes a single versioned archive of the fitted components.
Loading restores it. Loading reads pickled data, so only load checkpoints you
created yourself or that come from a source you trust. When no checkpoint
directory is given, training writes any intermediate checkpoints to a
temporary directory.

The steps for training, saving and loading are in
:doc:`/how-to/train_save_load`, and the choices made when tuning are in
:doc:`/how-to/tune_hyperparameters`.
