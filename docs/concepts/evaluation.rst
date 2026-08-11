Evaluation settings and metrics
===============================

DrEvalPy standardizes how models are split, scored, and stress-tested so
comparisons stay reproducible across studies and interfaces.

Cross-validation settings
-------------------------

Four leave-out modes control what is held out at validation and test time:

.. image:: /_static/img/LPO.png
    :width: 24%
    :alt: Leave-Pair-Out setting

.. image:: /_static/img/LCO.png
    :width: 24%
    :alt: Leave-Cell-Line-Out setting

.. image:: /_static/img/LTO.png
    :width: 24%
    :alt: Leave-Tissue-Out setting

.. image:: /_static/img/LDO.png
    :width: 24%
    :alt: Leave-Drug-Out setting

* **LPO (Leave-Pair-Out).** Random cell-line–drug pairs are held out, but both
  the drug and the cell line may already appear in training. Easiest setting;
  mainly useful to check whether a model can complete missing training
  entries.
* **LCO (Leave-Cell-Line-Out).** Entire cell lines are held out while drugs may
  overlap training. Relevant for personalized medicine: can the model predict
  a new cell line?
* **LTO (Leave-Tissue-Out).** Entire tissues are held out. Harder than LCO;
  relevant for tissue transfer and drug-repurposing style questions.
* **LDO (Leave-Drug-Out).** Entire drugs are held out while cell lines may
  overlap training. Usually the hardest setting; relevant for new-drug
  prediction in discovery workflows.

Baselines and naive predictors
------------------------------

Drug response values often have strong drug- or cell-line-specific means. A
model that only recovers those effects can look strong on naive correlation
or R²-style scores. We therefore always compare against naive predictors
(overall mean, cell-line mean, drug mean, tissue mean, tissue–drug mean, and
the ANOVA-style ``NaiveMeanEffectsPredictor``).

Baselines are tuned and scored alongside primary models, but randomization
and robustness stress tests apply to primary models only.
``NaiveMeanEffectsPredictor`` is required for normalized metrics and is added
when missing. It combines dataset, tissue (when available), cell-line, and drug
effects and is usually the strongest naive reference across settings.

Randomization modes
-------------------

Randomization asks how much performance drops when input views are scrambled
while the rest of the experiment stays fixed.

Modes:

* **SVCC** — Single View Constant (cell lines): one cell-line view stays
  intact; other cell-line views are perturbed.
* **SVRC** — Single View Random (cell lines): one cell-line view is
  randomized; others stay intact.
* **SVCD** — Single View Constant (drugs): one drug view stays intact; other
  drug views are perturbed.
* **SVRD** — Single View Random (drugs): one drug view is randomized; others
  stay intact.

Types:

* **permutation** — shuffle features across instances, preserving feature
  distributions but breaking the link to the target (default).
* **invariant** — scramble while preserving a key statistic (for matrices,
  mean and standard deviation per instance; for networks, degree
  distribution).

Robustness
----------

Robustness reruns training with varying random seeds. The number of trials
controls how many independent seeds you collect to judge run-to-run
stability.

Metrics
-------

Reported metrics include **MSE**, **RMSE**, **MAE**, **R²**, **Pearson**,
**Spearman**, and **Kendall**. Hyperparameter optimization defaults to
**RMSE**, not R².

Normalized metrics
~~~~~~~~~~~~~~~~~~

Drug- and cell-line-specific means can make ordinary R² and correlation look
strong even when a model adds little beyond those effects (Simpson's paradox
style inflation). The HTML report therefore also computes **normalized**
variants of R², Pearson, Spearman, and Kendall.

For each prediction row, DrEvalPy subtracts the corresponding
``NaiveMeanEffectsPredictor`` prediction from both the true response and the
model prediction, then recomputes those metrics on the residuals:

.. code-block:: text

   y_true_norm = y_true - y_pred_NaiveMeanEffects
   y_pred_norm = y_pred - y_pred_NaiveMeanEffects

MSE, RMSE, and MAE are **not** reported in normalized form — residualizing by
a strong mean-effects baseline mainly changes R² and the correlation metrics.
Normalized scores answer: “after removing what the mean-effects baseline
already explains, how much remaining structure does the model capture?”
Positive normalized R² / correlation means the model beats that baseline on
the residual signal; values near zero mean it mostly tracks mean effects.

Response transforms
-------------------

Optional response transforms applied before fitting:

* **none** — leave the measure unchanged
* **standard** — zero-mean / unit-variance scaling
* **minmax** — scale into a fixed range
* **robust** — scale using robust statistics (median / IQR-style)

Cross-study evaluation
----------------------

Cross-study evaluation trains on one dataset and scores on others that share
compatible identifiers and feature spaces. Prefer CurveCurator-refit measures
so the response definition itself is aligned across studies.

How to run experiments
----------------------

- :doc:`/cli/experiment` — CLI experiment settings
- :doc:`/cli/reporting` — HTML report after a run
- :doc:`/python/experiments` — ``mu_experiment`` options and results
- :doc:`/python/visualization` — ``evaluate``, plots, and ``create_report``
