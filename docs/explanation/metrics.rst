Why pooled metrics mislead and per-drug metrics help
====================================================

DrEvalPy reports seven metrics for every model: MSE, RMSE, MAE, :math:`R^2`,
Pearson correlation, Spearman correlation and Kendall correlation. The formulas,
their direction and the names used in result tables are listed in
:doc:`/reference/metrics`. This page explains why a single pooled score over all
test pairs can hide the thing you care about, and how DrEvalPy compensates.

The pooled score is dominated by the drug
-----------------------------------------

Suppose you pool every test pair and compute one correlation between true and
predicted responses. The variance of the true responses has two sources:

- *between-drug* variance: some drugs are potent against nearly every sample and
  others are not, so drug means differ widely;
- *within-drug* variance: for one drug, different samples respond differently.

In most drug screens the between-drug part is large. A predictor that knows only
each drug's average response captures all of it. Its pooled correlation is high,
even though it assigns exactly the same value to every sample treated with the
same drug and so tells you nothing about which samples will respond. This is a
Simpson-like effect: a trend that dominates the pooled data says little about
the trend inside each group.

``NaiveDrugMeanPredictor`` shows this directly. On a small synthetic dataset in
the leave-pair-out split mode, it reaches a pooled Pearson correlation of about
0.85, while its Pearson correlation within each drug is zero, because it predicts
a constant for each drug. The same holds for sample means within a sample. The
numbers depend on the dataset, but the pattern does not: a high pooled score
can come entirely from drug potency.

For personalized medicine the within-drug question is the one that matters. A
clinician choosing among samples for one drug needs the ranking of samples for
that drug, not the ranking of drugs.

Why DrEvalPy reports per-drug metrics
-------------------------------------

Computing the metric separately for each drug, and then summarizing across
drugs, removes the between-drug variance by construction. A drug-mean predictor
scores near zero per drug, and a model has to rank samples within each drug to
score well. The summary therefore includes per-drug metrics next to the pooled
ones, and the report plots per-drug correlations so that models can be compared
drug by drug.

Two details are worth knowing when you read per-drug numbers:

- A group needs at least two pairs, and a response that is not constant, to have a
  correlation. Other groups are skipped, for every model alike, so two models are always
  averaged over the same groups.
- :math:`R^2` has no lower bound, so a per-group :math:`R^2` is fragile on small groups.
  A group with two test pairs and a nearly flat response, with a range of 0.01, can score an
  :math:`R^2` tens of thousands below zero, and one such group can dominate the mean over
  groups. DrEvalPy keeps the metric as it is. Read the per-group correlations, or the number
  of groups behind each score, alongside the per-group :math:`R^2`.
- A predictor that is constant within a group scores 0 there, because a constant
  prediction carries no ranking information. This is why the drug-mean baseline scores
  zero per drug. In the per-group comparison, a model that predicts a constant for every
  group, such as the naive drug and sample means, is drawn in the scatter plots at zero.

The first tutorial, :doc:`/tutorials/first_benchmark`, shows
these numbers for a first benchmark.

Why DrEvalPy compares against NaiveMeanEffectsPredictor
-------------------------------------------------------

Per-drug scores answer one confound. A related confound is that samples also
differ in their average sensitivity, and tissues differ too. A model can look good
within a drug simply because it knows which samples are generally sensitive.

``NaiveMeanEffectsPredictor`` captures the dataset mean, the tissue effect, the
sample effect and the drug effect additively (see :doc:`evaluation_design`). It
is the strongest simple reference that uses no features. So that "better than
this" becomes a precise statement, DrEvalPy can normalize a result against it.

For each pair, the reference prediction :math:`\hat{y}^{\mathrm{ref}}` is
subtracted from both the true response and the model prediction:

.. math::

   y^{\mathrm{norm}} = y - \hat{y}^{\mathrm{ref}}, \qquad
   \hat{y}^{\mathrm{norm}} = \hat{y} - \hat{y}^{\mathrm{ref}}

All seven metrics are then recomputed on these residuals. The normalized scores
answer: after removing what the mean-effects reference already explains, how
much of the remaining structure does the model capture? A positive normalized
:math:`R^2` or correlation means the model beats the reference on the residual
signal. A value near zero means the model mostly tracks mean effects.

Three consequences follow from the definition:

- MSE, RMSE and MAE do not change under normalization, because the same
  reference value is subtracted from the truth and the prediction, so their
  difference is unchanged. Normalization matters for :math:`R^2` and the
  correlations.
- The reference model itself is dropped from the normalized result, since its
  residual is zero by construction. Only the leaderboard and the critical
  difference diagram of the report still list it, as the baseline that the other
  models are ranked against.
- Normalization is explicit. ``drevalpy.run`` never adds the reference model for
  you: it must be among the models you ran, and normalization fails with a clear
  error if it is missing, if the result is already normalized, or if the
  reference has more than one ordinary run. The report normalizes only when you
  ask for a reference model. The step-by-step version is in
  :doc:`/how-to/make_report`.

Putting it together
-------------------

Read the scores in this order:

1. Pooled scores tell you whether the model reproduces the overall pattern. They
   are easy to inflate, so treat them as a sanity check.
2. Per-drug scores tell you whether the model ranks samples within a drug.
3. Normalized scores tell you whether the model adds anything beyond the
   additive sample, tissue and drug averages.
4. Stress tests (see :doc:`evaluation_design`) tell you whether the features were
   the source of the signal.

A model that scores well on the first and poorly on the second and third has
learned to rank drugs and samples by their averages. It might still be useful, but
the features are not why.
