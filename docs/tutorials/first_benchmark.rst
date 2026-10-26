Benchmark your first models
===========================

.. interface: both

In this tutorial you evaluate three models on a small simulated dataset and learn to read
the result. You end up with a results directory, a report, and one lesson that applies to
every drug response evaluation: a high score over all pairs can come entirely from knowing
which drugs are potent.

The dataset is simulated, so nothing is downloaded and everything runs offline in about a
minute. You need DrEvalPy installed (see :doc:`/how-to/install`). Pick the **CLI** or the
**Python** tab on any block below; the page remembers your choice.

Create a dataset
----------------

The simulated dataset has 60 samples, 8 drugs and a ``gene_expression`` view of about 11,600 gene
features per sample. About 5 % of the sample-drug combinations are missing, as in a real screen.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy data synthetic --output toy.h5mu

      The command writes the dataset to ``toy.h5mu`` and prints:

      .. code-block:: text

         Wrote SYNTHETIC to toy.h5mu

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import drevalpy

         dataset = drevalpy.synthetic()
         print(dataset)

      The call returns the dataset in memory and prints:

      .. code-block:: text

         Dataset
             Name: SYNTHETIC
             Samples: 60
             Drugs: 8
             Measured pairs: 456
             Randomization: None
             Modalities:
                 gene_expression: 60 × 11650

A pair is one sample-drug combination with a measured response. The dataset holds 456 of the
60 × 8 = 480 possible pairs. Drugs differ strongly in average potency, and hidden sample factors drive both
the gene expression and the responses. That is the
structure a real screen has, in a size you can train on in seconds. The generator is
``drevalpy.synthetic``, and ``--seed`` changes the draw; see :doc:`/reference/cli`.

Choose three models
-------------------

An experiment needs reference points. You use two naive baselines, which use no features beyond
the identities of samples and drugs, and one model that uses features:

``NaiveDrugMeanPredictor``
   Predicts the mean response of the pair's drug in the training pairs. It knows nothing about
   samples.

``NaiveMeanEffectsPredictor``
   Predicts the training mean plus an effect for the pair's tissue, sample and drug. It is the
   reference that other models have to beat.

``ElasticNet``
   A linear model on the scaled gene expression of the sample and the Morgan fingerprint of the
   drug. Its recipe is ``scaledGeneExpression:fingerprints:elasticNet``; see
   :doc:`/reference/recipes`.

All three are zoo presets, selected by their zoo name. :doc:`/how-to/list_available` shows how
to list the others.

Run the experiment
------------------

An experiment evaluates the models over all folds of one split mode. You use leave-pair-out
(LPO), which holds out random pairs, and 5 folds, which ``run`` always uses.

``ElasticNet`` has hyperparameters, so DrEvalPy tunes them on a validation part of each
training set before the final fit. ``--hpo-trials 8`` tries 8 configurations per fold; the
default is 16. Keep tuning on for ``ElasticNet``: with its default hyperparameters it already
learns, but tuning lowers its RMSE over all pairs from 0.778 to 0.746 on this dataset.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy run NaiveDrugMeanPredictor NaiveMeanEffectsPredictor ElasticNet \
             --dataset toy.h5mu --split-mode LPO --hpo-trials 8 --group-by drug

      Tuning logs one line per trial and fold while it runs. The command ends with this
      output:

      .. code-block:: text

         Wrote experiment results to results
         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 3
                 NaiveDrugMeanPredictor (5 folds): MSE=0.7908, RMSE=0.8880, MAE=0.7195, R^2=0.7031, Pearson=0.8415, Spearman=0.8440, Kendall=0.6882
                 NaiveMeanEffectsPredictor (5 folds): MSE=0.8399, RMSE=0.9144, MAE=0.7218, R^2=0.6851, Pearson=0.8298, Spearman=0.8409, Kendall=0.6466
                 ElasticNet (5 folds): MSE=0.5569, RMSE=0.7458, MAE=0.5860, R^2=0.7908, Pearson=0.8926, Spearman=0.8970, Kendall=0.7189
         Metrics computed within each drug, averaged over drugs and folds (number of drugs scored in brackets):
         Model                      MSE        RMSE       MAE        R^2         Pearson    Spearman   Kendall
         NaiveDrugMeanPredictor     0.789 (8)  0.866 (8)  0.720 (8)  -0.124 (8)  0.000 (8)  0.000 (8)  0.000 (8)
         NaiveMeanEffectsPredictor  0.852 (8)  0.895 (8)  0.731 (8)  -0.583 (8)  0.386 (8)  0.372 (8)  0.277 (8)
         ElasticNet                 0.564 (8)  0.727 (8)  0.593 (8)  -0.018 (8)  0.520 (8)  0.480 (8)  0.359 (8)

      The results are in the ``results`` directory, one subdirectory per model; see
      :doc:`/reference/results`.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy import ModelSpec

         names = ["NaiveDrugMeanPredictor", "NaiveMeanEffectsPredictor", "ElasticNet"]
         models = [ModelSpec.from_name(name) for name in names]
         results = drevalpy.run(models, dataset, "LPO", hpo_num_trials=8)
         print(results)

      Tuning logs one line per trial and fold while it runs. The ``print`` shows:

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 3
                 NaiveDrugMeanPredictor (5 folds): MSE=0.7908, RMSE=0.8880, MAE=0.7195, R^2=0.7031, Pearson=0.8415, Spearman=0.8440, Kendall=0.6882
                 NaiveMeanEffectsPredictor (5 folds): MSE=0.8399, RMSE=0.9144, MAE=0.7218, R^2=0.6851, Pearson=0.8298, Spearman=0.8409, Kendall=0.6466
                 ElasticNet (5 folds): MSE=0.5569, RMSE=0.7458, MAE=0.5860, R^2=0.7908, Pearson=0.8926, Spearman=0.8970, Kendall=0.7189

      ``results`` is an ``ExperimentResult``; ``results.save("results")`` writes it to a
      directory, as the CLI does.

Read the scores over all pairs
------------------------------

The first block of numbers pools the held-out pairs of each fold, computes each metric once
per fold and averages over the 5 folds. Look at the Pearson correlation between measured and
predicted response:

========================= ===================
Model                     Pearson (all pairs)
========================= ===================
NaiveDrugMeanPredictor    0.842
NaiveMeanEffectsPredictor 0.830
ElasticNet                0.893
========================= ===================

Every model scores above 0.8, and the model that knows only each drug's average, with no idea
which sample it is predicting for, scores 0.84. ``ElasticNet`` leads by about 0.05. The pooled
numbers hardly separate a model that uses gene expression from one that does not.

Read the scores within each drug
--------------------------------

The drugs differ so much in potency that the drug explains most of the variance between all
pairs. A predictor that gets the drug right gets a high pooled score, whatever it does with the
samples. The scientific question is different: for one drug, which samples respond? To answer it,
compute each metric within each drug and average over the drugs and folds.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      ``--group-by drug`` already printed this table at the end of the output above. ``--group-by
      sample`` does the same within each sample, and ``drevalpy aggregate`` accepts the flag too.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         per_drug = results.summary_table(group_by="drug")
         print(per_drug[["Pearson", "Spearman", "RMSE", "R^2"]].round(3))

      The output:

      .. code-block:: text

                                    Pearson  Spearman   RMSE    R^2
         NaiveDrugMeanPredictor       0.000     0.000  0.866 -0.124
         NaiveMeanEffectsPredictor    0.386     0.372  0.895 -0.583
         ElasticNet                   0.520     0.480  0.727 -0.018

      ``summary_group_counts(group_by="drug")`` gives the number of drugs behind each cell.

The Pearson column changes completely:

========================= =========================== ======================
Model                     Pearson (all pairs)         Pearson (within drug)
========================= =========================== ======================
NaiveDrugMeanPredictor    0.842                       0.000
NaiveMeanEffectsPredictor 0.830                       0.386
ElasticNet                0.893                       0.520
========================= =========================== ======================

``NaiveDrugMeanPredictor`` predicts one value per drug. Within a drug its predictions are
constant and carry no ranking information, so its correlation is 0 in every drug, and all 8
drugs are scored. The other two models rank samples within each drug. The gap
between ``ElasticNet`` and the reference ``NaiveMeanEffectsPredictor`` is 0.06 over all pairs and
0.13 within drugs, so within drugs the gene expression is visibly worth something. The within-drug
:math:`R^2` of every model is at or below 0: on this small dataset no model predicts a drug's
test pairs better than their own mean does.

This pattern, a high pooled score that hides the within-drug ranking, is what :doc:`/explanation/metrics`
explains. :doc:`/reference/metrics` lists the metrics and how they are summarized.

Build a report
--------------

The report gathers the tables and plots of an experiment in one HTML file.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy report results --output-dir report

      The command prints the progress of the plots and ends with:

      .. code-block:: text

         Report generated at report

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy.visualization.report import create_report

         create_report(results, "report")

Open ``report/Drug-Response-Evaluation_multiqc_report.html`` in a browser. These sections
matter most for this experiment:

Model Summary Table
   The pooled metrics, mean and standard deviation over the folds. These are the numbers
   from the first block above.

Model Summary Table per Drug
   The metrics within each drug, with the number of drugs behind every cell. This is the table you
   read above. An empty cell is an undefined score: no group could be scored, which depends on the
   data and is the same for every model.

Performance Heatmap and Metric Distributions
   All metrics of all models in one heatmap, and their spread over the folds.

Regression density, Drug-wise comparison, Sample-wise comparison
   Prediction against measurement for each model, and the models compared pairwise by their
   correlation within each drug and each sample.

The report normalizes against a reference model only if you pass one (``--reference-model`` on
the command line, ``reference_model=`` in Python); see :doc:`/how-to/make_report`.

What you learned
----------------

- ``drevalpy data synthetic`` and ``drevalpy.synthetic()`` give an offline dataset that behaves
  like a real screen.
- ``drevalpy run`` and ``drevalpy.run`` evaluate several models over the folds of one split mode.
- A pooled score is dominated by drug potency. Read the within-drug scores, and compare a model
  with ``NaiveMeanEffectsPredictor``, not with nothing.

Next, :doc:`own_predictor` builds a predictor of your own and benchmarks it against these baselines,
and :doc:`tune_and_stress_test` reads tuning and stress-test results. Why LPO is only one of four
questions you can ask is explained in :doc:`/explanation/evaluation_design`, and the split modes are
listed in :doc:`/reference/split_modes`.
