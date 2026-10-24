Tune a model and stress-test it
===============================

.. interface: both

In this tutorial you tune ``ElasticNet``, look at what the tuning tried, and then test how much the
model depends on its gene expression features and on chance. You run one experiment that includes a
randomization test and a robustness test, and read both. Along the way you see why the score over
all pairs is the wrong place to look.

You need the skills of :doc:`first_benchmark`. The dataset is simulated, so everything runs offline.
The experiment takes about 20 seconds.

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
   A randomization test. For each sample view of a model, ``SVRC`` permutes that view across the
   samples and trains the model again on the damaged data. ``ElasticNet`` has one sample view,
   ``gene_expression``, so it gets one extra run per fold. A model without views, such as
   ``NaiveMeanEffectsPredictor``, gets none.

``--robustness-trials 2`` (``robustness_trials=2``)
   A robustness test. Each fold is repeated twice more, with a new seed for the pair order and for
   the model's own randomness. The spread of the repeats shows how much the score depends on chance.

All three belong to the experiment, so it trains ``ElasticNet`` on 5 folds × 3 splits (the
original and two repeats) × 2 datasets (the original and the permuted one) = 30 times.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy run NaiveMeanEffectsPredictor ElasticNet \
             --dataset toy.h5mu --split-mode LPO --hpo-trials 4 \
             --randomization-mode SVRC --robustness-trials 2

      The command ends with:

      .. code-block:: text

         Wrote experiment results to results
         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 2
                 NaiveMeanEffectsPredictor (5 folds): MSE=0.8399, RMSE=0.9144, MAE=0.7218, R^2=0.6851, Pearson=0.8298, Spearman=0.8409, Kendall=0.6466
                 ElasticNet (5 folds): MSE=0.5495, RMSE=0.7407, MAE=0.5832, R^2=0.7939, Pearson=0.8935, Spearman=0.9024, Kendall=0.7275

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy import ModelSpec

         names = ["NaiveMeanEffectsPredictor", "ElasticNet"]
         models = [ModelSpec.from_name(name) for name in names]
         results = drevalpy.run(
             models,
             dataset,
             "LPO",
             hpo_num_trials=4,
             randomization_modes=["SVRC"],
             robustness_trials=2,
         )
         print(results)

      The output:

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 2
                 NaiveMeanEffectsPredictor (5 folds): MSE=0.8399, RMSE=0.9144, MAE=0.7218, R^2=0.6851, Pearson=0.8298, Spearman=0.8409, Kendall=0.6466
                 ElasticNet (5 folds): MSE=0.5495, RMSE=0.7407, MAE=0.5832, R^2=0.7939, Pearson=0.8935, Spearman=0.9024, Kendall=0.7275

The summary has the same numbers as in the first tutorial. It averages only the ordinary runs, the
ones on the original data in the original pair order, so the stress tests do not change it.
The stress-test runs are in the saved results, and no command prints them. The next two sections
read them in Python.

Read the tuning
---------------

Every run keeps the trials of its tuning. Look at fold 0 of ``ElasticNet``.

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
             print(number, round(trial.score, 3), round(trial.hyperparameters["predictor.elasticNet.alpha"], 4))
         best = fold_0.best_hyperparameters
         print("best:", {key: round(best[key], 4) for key in ("alpha", "l1_ratio")})

   .. tab-item:: Python
      :sync: python

      The ``results`` object returned by ``drevalpy.run`` already holds them:

      .. code-block:: python

         experiment = results
         elastic = next(model for model in experiment.models if model.model_name == "ElasticNet")
         fold_0 = elastic.select_runs()[0]
         for number, trial in enumerate(fold_0.trials):
             print(number, round(trial.score, 3), round(trial.hyperparameters["predictor.elasticNet.alpha"], 4))
         best = fold_0.best_hyperparameters
         print("best:", {key: round(best[key], 4) for key in ("alpha", "l1_ratio")})

Both print the same lines:

.. code-block:: text

   0 0.637 0.0002
   1 0.636 0.0008
   2 0.634 0.0029
   3 0.623 0.1091
   best: {'alpha': 0.1091, 'l1_ratio': 0.1705}

Each line is one trial: its number, its validation RMSE (the tuning metric, lower is better) and
the penalty strength ``alpha`` it tried. Trial 3 had the lowest RMSE, so the final model of fold 0
uses ``alpha`` = 0.1091. Each fold tunes on its own validation pairs, so other folds can pick other
configurations. The scores differ little between these four trials, but a strong penalty can
ruin the model: a search with 8 trials also tries ``alpha`` = 1.39 in this fold, and its RMSE is
1.14. The default ``alpha`` is 0.1, close to the winner here, so tuning refines a usable preset
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
             "robustness trial 0": elastic.select_runs(robustness=0),
             "robustness trial 1": elastic.select_runs(robustness=1),
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
             "robustness trial 0": elastic.select_runs(robustness=0),
             "robustness trial 1": elastic.select_runs(robustness=1),
         }
         print(pd.DataFrame({name: pearson_scores(runs) for name, runs in settings.items()}).T.round(3))

Both print:

.. code-block:: text

                             all pairs  within drug
   ordinary                      0.893        0.513
   gene_expression permuted      0.844        0.152
   robustness trial 0            0.893        0.513
   robustness trial 1            0.893        0.513

Two things stand out.

**The randomization test.** Permuting ``gene_expression`` across the samples removes what the
features say about each sample, and the within-drug score falls from 0.51 to 0.15. The score over
all pairs falls only from 0.89 to 0.84: the drug fingerprints still tell the model which drug it is
predicting, and that carries the pooled score. This is the pattern of the first tutorial again.
The test shows that ``ElasticNet`` uses its gene expression, and that the within-drug score is the
one that registers it. Some within-drug score remains. A likely reason is that leave-pair-out leaves
every test sample in the training set, where its permuted features still work as an identifier of
that sample.

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

- ``--hpo-trials`` sets the number of tuning trials. Every run keeps its trials, so you can see
  what the search tried.
- A randomization test retrains on a permuted view and shows how much a model relies on it. Read
  it within drugs.
- A robustness test repeats a fold with new seeds and shows how much a score depends on chance.
- The summary table averages ordinary runs. Stress-test runs live in the saved
  ``ExperimentResult``, and you read them with ``select_runs``.

To run the same experiment on your own data, start from :doc:`/how-to/bring_your_own_data`. For the
other split modes, see :doc:`/explanation/evaluation_design`.
