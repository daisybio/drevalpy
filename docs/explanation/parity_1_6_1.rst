How DrEvalPy 2 was checked against 1.6.1
========================================

DrEvalPy 2 rewrites every model of DrEvalPy 1.6.1 as a composition of featurizers
and a predictor. A rewrite can change results without anyone noticing, so each 1.6.1
model was run side by side with its 2.0 counterpart on the same inputs, and the
outputs were compared step by step. This page explains how that comparison works,
what its verdicts mean, and where 2.0 deliberately gives different results. The
per-model changes to the literature models are in :doc:`literature_models`; the
mapping from 1.6.1 model names to 2.0 presets is in :doc:`/how-to/migrate/index`.

What is compared
----------------

The comparison harness lives in ``tools/v1_parity/`` in the repository. It is not part
of the installed package and does not run in continuous integration. For each model it:

- draws a small synthetic case from a fixed seed: samples, drugs, gene expression and
  other views, drug fingerprints or SMILES, a response matrix and a train/test split;
- runs the 1.6.1 code and the 2.0 code in two separate Python environments, because
  both packages are called ``drevalpy``;
- records not only the predictions but the intermediate steps: the selected genes,
  the fitted scalers, the design matrix that reaches the estimator, the fitted
  coefficients or importances, and for neural networks the forward pass with the
  1.6.1 weights copied into the 2.0 network;
- compares every step under a tolerance class that is fixed before the run.

Where an estimator could hide a difference in its inputs, the harness also refits
both sides' estimators on one shared design matrix. If they then agree exactly, any
remaining difference comes from the features, not from the estimator.

The results are committed as one JSON file per model in ``tools/v1_parity/evidence/``,
with the versions of Python, NumPy, PyTorch and scikit-learn on both sides and the
commit they were produced from.

Tolerance classes and verdicts
------------------------------

Each step declares one of four classes, from strictest to loosest:

``exact``
   Equal to within 1e-12. Used for gene selection and order, scalers, response
   transformations, the naive predictors, and seeded forests, nearest-neighbour and
   linear models.
``forward``
   Equal to within 1e-6. Used for neural forward passes with copied weights, and for
   any step whose input is held as 32-bit floats in 2.0 (see below).
``train``
   Equal to within a relative 1e-4. Used for short seeded training runs on the CPU
   where both sides draw random numbers in the same order.
``statistical``
   A Pearson correlation of at least 0.99 and a mean difference of at most a tenth of a
   standard deviation. Used where the order of random draws cannot be matched, such as
   data loader shuffles.

A step is either a *parity* step, which must meet its class, or a *deviation*, a
known and explained difference whose size is recorded but which does not count
against the model. A model's verdict is ``mismatch`` if any parity step misses its
class, and otherwise the loosest class among its parity steps.

The verdicts
------------

No model ends with ``mismatch``.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - 1.6.1 model
     - Verdict
     - Note
   * - NaivePredictor, NaiveDrugMeanPredictor, NaiveCellLineMeanPredictor,
       NaiveTissueMeanPredictor, NaiveTissueDrugMeanPredictor,
       NaiveMeanEffectsPredictor
     - exact
     - NaiveCellLineMeanPredictor is ``NaiveSampleMeanPredictor`` in 2.0.
   * - AdaBoostDecisionTree, GradientBoosting, KNNRegressor, Lasso, RandomForest,
       SingleDrugElasticNet, SingleDrugRandomForest, SingleDrugKNN
     - forward
     - Exact on a shared design matrix; ``forward`` because 2.0 features are 32-bit.
   * - ElasticNet
     - train
     - 2.0 solves the ridge case in 32-bit floats, so its predictions agree within
       ``train``.
   * - SVR
     - statistical
     - The libsvm solver stops at a tolerance, and the 32-bit inputs lead it to a
       different, equally valid solution.
   * - MultiViewXGBoost, MultiViewLightGBM
     - forward
     -
   * - MultiViewRandomForest
     - statistical
     - 1.6.1 never passed the drug features to the forest; see below.
   * - SimpleNeuralNetwork, DrugGNN, SRMF, MOLIR, SuperFELTR, PharmaFormer, DIPK,
       SparseGO, EnsembleMF, PaccMann, RandomForestAEMolGNet
     - train
     -
   * - Precily
     - forward
     -

Three further cases check shared code rather than a model: the response
transformations (``exact``), the leaderboard and its per-drug scores (``exact``), and
the sklearn models on the proteomics view (``forward``).

Differences kept on purpose
---------------------------

Some 1.6.1 behaviour was not carried over, because it was a bug or because 2.0
evaluates more strictly. Each of these is recorded as a deviation in the evidence:

- **32-bit features.** 2.0 stores views as 32-bit floats, where 1.6.1 computed most
  features in 64-bit. Predictions agree to about 1e-6. For tree models, a rounding
  difference this small can occasionally move a sample to the other side of a split:
  a random forest on the proteomics view then predicts up to 0.05 differently.
- **Fitting on training data only.** Precily computes its GSVA scores per fold, and
  PharmaFormer learns its drug vocabulary from the training drugs only. 1.6.1 computed
  both once on the whole dataset, test samples and drugs included.
- **One featurizer per model.** The single-drug models and MOLIR and SuperFELTR fit
  their scaler and gene selection once on all training samples, not once per drug.
- **Gene selection in MOLIR.** 1.6.1 ranked genes by variance after standardizing
  them, when every gene has a variance of about one, so the choice came down to
  rounding. 2.0 ranks before standardizing.
- **Unseen samples and drugs in SRMF.** 2.0 fits the factors on training entities only
  and predicts the training mean for an entity it has not seen, where 1.6.1 built the
  factors over every sample in the feature table.
- **Best epoch.** DrugGNN restores the weights of its best validation epoch; 1.6.1 kept
  the last epoch.
- **MOLIR loss.** 2.0 weights the triplet term of the loss by ``gamma``; 1.6.1 did not.
- **EnsembleMF.** A pair without features is predicted as missing, not as a number.
- **MultiViewRandomForest.** 1.6.1 built the design matrix from the sample views only,
  so it predicted the same value for a sample with every drug. 2.0 includes the drug
  fingerprints.
- **Methylation PCA.** 1.6.1 standardized methylation before the PCA in the boosted
  tree models; the 2.0 ``pca`` featurizer has no scaler in front of it.
- **Edge cases that failed in 1.6.1.** A molecule without bonds in DrugGNN, and a last
  prediction batch of one row in SparseGO.

Random number streams that changed between library versions, for example in
XGBoost's row and column sampling, are recorded as deviations too. They are a
property of the installed library, not of DrEvalPy.

Limits of the comparison
------------------------

The cases are small and synthetic, so that the whole comparison runs on a CPU. They exercise every code path that turns inputs into a prediction, but they do
not reproduce the 1.6.1 benchmark results on GDSC or CTRP; scores on the real datasets
can move slightly for the reasons above. The comparison also covers the default
hyperparameters and a few chosen variants, not every point of a search grid.
