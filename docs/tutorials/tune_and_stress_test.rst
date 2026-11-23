Tune a model and stress-test it
===============================

.. interface: both

In this tutorial you tune ``ElasticNet``, look at what the tuning tried, and then test how much the
model depends on its gene expression features and on chance. You run one experiment that includes a
randomization test and a robustness test, and read both. Along the way you see why the score over
all pairs is the wrong place to look.

You need the skills of :doc:`first_benchmark`. The dataset is simulated, so everything runs offline.
The experiment takes about 10 seconds: it uses 2 folds instead of the default 5.

Create a dataset
----------------

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy data synthetic --output toy.h5mu

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import drevalpy

         dataset = drevalpy.synthetic()

The dataset is the one from the first tutorial: 60 samples, 8 drugs and a ``gene_expression`` view.

Run with tuning and stress tests
--------------------------------

Three options turn a plain experiment into a tuned, stress-tested one:

``--hpo-trials 4`` (``hpo_num_trials=4``)
   Tries 4 hyperparameter configurations, called trials, on a validation part of each training
   set, and trains the final model with the best one. Tuning is on by default, with 16 trials.

``--randomization-mode SVRC`` (``randomization_modes=["SVRC"]``)
   A randomization test. For each sample view of a model, ``SVRC`` permutes the features the model
   gets from that view across the samples and trains the model again on the damaged features. ``ElasticNet`` has one sample view,
   ``gene_expression``, so it gets one extra run per fold. A model without views, such as
   ``NaiveMeanEffectsPredictor``, gets none.

``--robustness-repeats 2`` (``robustness_repeats=2``)
   A robustness test. Each fold is repeated twice more, with a new seed for the pair order and for
   the model's own randomness. The spread of the repeats shows how much the score depends on chance.

All three belong to the experiment, so it trains ``ElasticNet`` on 2 folds × 3 splits (the
original and two repeats) × 2 datasets (the original and the permuted one) = 12 times. ``--n-splits 2``
(``n_splits=2``) keeps that short; the default is 5 folds.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy experiment NaiveMeanEffectsPredictor ElasticNet \
             --dataset toy.h5mu --split-mode LPO --n-splits 2 --hpo-trials 4 \
             --randomization-mode SVRC --robustness-repeats 2

      The command ends with:

      .. code-block:: text

         Wrote experiment results to results
         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 2
                 NaiveMeanEffectsPredictor (2 folds): MSE=6.7874, RMSE=2.6038, MAE=2.0723, R^2=0.5215, Pearson=0.7443, Spearman=0.7519, Kendall=0.5504
                 ElasticNet (2 folds): MSE=3.2995, RMSE=1.8164, MAE=1.4247, R^2=0.7683, Pearson=0.8816, Spearman=0.8911, Kendall=0.7086

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy import ModelSpec

         names = ["NaiveMeanEffectsPredictor", "ElasticNet"]
         models = [ModelSpec.from_name(name) for name in names]
         results = drevalpy.experiment(
             models,
             dataset,
             "LPO",
             n_splits=2,
             hpo_num_trials=4,
             randomization_modes=["SVRC"],
             robustness_repeats=2,
         )
         print(results)

      The output:

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 2
                 NaiveMeanEffectsPredictor (2 folds): MSE=6.7874, RMSE=2.6038, MAE=2.0723, R^2=0.5215, Pearson=0.7443, Spearman=0.7519, Kendall=0.5504
                 ElasticNet (2 folds): MSE=3.2995, RMSE=1.8164, MAE=1.4247, R^2=0.7683, Pearson=0.8816, Spearman=0.8911, Kendall=0.7086

The summary is the one the same command prints without the two stress-test options. It averages
only the ordinary runs, the ones on the original data in the original pair order, so the stress
tests do not change it.
The stress-test runs are in the saved results, and no command prints them. The next two sections
read them in Python.

Read the tuning
---------------

Only the ordinary runs tune, and each keeps the trials of its tuning. The stress-test runs of a
fold reuse the best hyperparameters of its ordinary run and have no trials; see
:doc:`/how-to/run_stress_tests`. Look at fold 0 of ``ElasticNet``.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      Load the saved results directory and print the trials:

      .. code-block:: python

         from drevalpy.types.results import ExperimentResult

         experiment = ExperimentResult.load("results")
         elastic = next(model for model in experiment.models if model.model_name == "ElasticNet")
         fold_0 = elastic.select_runs()[0]
         for number, trial in enumerate(fold_0.trials):
             print(number, round(trial.metrics[trial.optimization_metric], 3), round(trial.hyperparameters["predictor.elasticNet.alpha"], 4))
         best = fold_0.best_hyperparameters
         print("best:", {key: round(best[f"predictor.elasticNet.{key}"], 4) for key in ("alpha", "l1_ratio")})

   .. tab-item:: Python
      :sync: python

      The ``results`` object returned by ``drevalpy.experiment`` already holds them:

      .. code-block:: python

         experiment = results
         elastic = next(model for model in experiment.models if model.model_name == "ElasticNet")
         fold_0 = elastic.select_runs()[0]
         for number, trial in enumerate(fold_0.trials):
             print(number, round(trial.metrics[trial.optimization_metric], 3), round(trial.hyperparameters["predictor.elasticNet.alpha"], 4))
         best = fold_0.best_hyperparameters
         print("best:", {key: round(best[f"predictor.elasticNet.{key}"], 4) for key in ("alpha", "l1_ratio")})

Both print the same lines:

.. code-block:: text

   0 1.882 0.0002
   1 1.887 0.0008
   2 1.879 0.0029
   3 1.788 0.1091
   best: {'alpha': 0.1091, 'l1_ratio': 0.1705}

Each line is one trial: its number, its validation RMSE (the tuning metric, lower is better) and
the penalty strength ``alpha`` it tried. Trial 3 had the lowest RMSE, so the final model of fold 0
uses ``alpha`` = 0.1091. Each fold tunes on its own validation pairs, so other folds can pick other
configurations. The three trials with a weak penalty score about the same, and the stronger
penalty of trial 3 lowers the RMSE by 0.09. Too strong a penalty hurts again, though: a search
with 8 trials also tries ``alpha`` = 1.39 in this fold, and its RMSE is 1.858. The preset starts from
``alpha`` = 1 and ``l1_ratio`` = 0 and already works: without tuning, this experiment scores an RMSE of
1.921 over all pairs, against 1.816 tuned, so tuning refines a usable preset
rather than rescuing it. See
:doc:`/how-to/tune_hyperparameters` for the metric, the number of trials and the search spaces, and
:doc:`/reference/hyperparameters` for the keys.

Read the stress tests
---------------------

A stress test compares the damaged runs with the ordinary ones. Compare the Pearson correlation of
four sets of ``ElasticNet`` runs, over all pairs and within each drug. The helper below computes the
within-drug score of a run: it correlates truth and prediction inside each drug and averages over
the drugs.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The ``experiment`` and ``elastic`` objects come from the code in the previous section.

      .. code-block:: python

         import numpy as np
         import pandas as pd


         def pearson_scores(runs):
             per_drug = []
             for run in runs:
                 frame = pd.DataFrame({"drug": run.drug_ids, "truth": run.ground_truth, "pred": run.predictions})
                 per_drug.append(frame.groupby("drug").apply(lambda g: g["truth"].corr(g["pred"]), include_groups=False).mean())
             pooled = np.mean([run.metrics["Pearson"] for run in runs])
             return {"all pairs": pooled, "within drug": np.mean(per_drug)}


         settings = {
             "ordinary": elastic.select_runs(),
             "gene_expression permuted": elastic.select_runs(randomization=("SVRC", "gene_expression")),
             "robustness repeat 0": elastic.select_runs(robustness=0),
             "robustness repeat 1": elastic.select_runs(robustness=1),
         }
         print(pd.DataFrame({name: pearson_scores(runs) for name, runs in settings.items()}).T.round(3))

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import numpy as np
         import pandas as pd


         def pearson_scores(runs):
             per_drug = []
             for run in runs:
                 frame = pd.DataFrame({"drug": run.drug_ids, "truth": run.ground_truth, "pred": run.predictions})
                 per_drug.append(frame.groupby("drug").apply(lambda g: g["truth"].corr(g["pred"]), include_groups=False).mean())
             pooled = np.mean([run.metrics["Pearson"] for run in runs])
             return {"all pairs": pooled, "within drug": np.mean(per_drug)}


         settings = {
             "ordinary": elastic.select_runs(),
             "gene_expression permuted": elastic.select_runs(randomization=("SVRC", "gene_expression")),
             "robustness repeat 0": elastic.select_runs(robustness=0),
             "robustness repeat 1": elastic.select_runs(robustness=1),
         }
         print(pd.DataFrame({name: pearson_scores(runs) for name, runs in settings.items()}).T.round(3))

Both print:

.. code-block:: text

                             all pairs  within drug
   ordinary                      0.882        0.477
   gene_expression permuted      0.874        0.425
   robustness repeat 0           0.882        0.477
   robustness repeat 1           0.882        0.477

Two things stand out.

**The randomization test.** Permuting ``gene_expression`` across the samples removes what the
features say about each sample, and the within-drug score falls from 0.477 to 0.425. The score over
all pairs falls only from 0.882 to 0.874: the drug fingerprints still tell the model which drug it
is predicting, and that carries the pooled score. This is the pattern of the first tutorial again.
The test shows that ``ElasticNet`` uses its gene expression, and that the within-drug score is the
one that registers it. Most of the within-drug score remains. A likely reason is that leave-pair-out
leaves every test sample in the training set, where its permuted features still work as an
identifier of that sample.

**The robustness test.** Both repeats score the same as the ordinary runs to three decimals.
``ElasticNet`` is deterministic once its hyperparameters are fixed, and the order of the pairs
makes no measurable difference to its fit, so there is no spread to see. A model with random components, such as
``RandomForest``, would show one. If a score depended on chance, the repeats would spread, and a
difference between two models smaller than that spread would mean nothing.

:doc:`/reference/split_modes` lists the randomization modes, their types and the exact meaning of
each stress test, and :doc:`/how-to/run_stress_tests` shows the options and how to choose the
modes. :doc:`/explanation/reproducibility` explains how the seeds are derived.

What you learned
----------------

- ``--hpo-trials`` sets the number of tuning trials. Every tuned run keeps its trials, so you can
  see what the search tried.
- A randomization test retrains on a permuted view and shows how much a model relies on it. Read
  it within drugs.
- A robustness test repeats a fold with new seeds and shows how much a score depends on chance.
- The summary table averages ordinary runs. Stress-test runs live in the saved
  ``ExperimentResult``, and you read them with ``select_runs``.

To run the same experiment on your own data, start from :doc:`/how-to/bring_your_own_data`. For the
other split modes, see :doc:`/explanation/evaluation_design`.
