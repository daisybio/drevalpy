Run stress tests
================

.. interface: both

A stress test checks whether a score means what you think it means. A **randomization test**
trains a model on data in which some of its features are shuffled; a model that really uses
those features must get worse. A **robustness test** repeats each fold with the order of the
pairs shuffled; a score that moves a lot between the repeats depends on chance. This page shows
how to run both, read the runs, and run them as separate jobs. :doc:`/tutorials/tune_and_stress_test`
reads the results of both tests in detail, and :doc:`/explanation/evaluation_design` explains
why they are worth the cost.

The examples use the simulated dataset and the ``Ridge`` zoo preset, which fits without tuning
and so keeps the example fast.

Run both tests
--------------

Add the randomization mode with ``--randomization-mode`` and the number of robustness repeats with
``--robustness-repeats``. The tests belong to the experiment: a single ``run`` trains the
model on the original data and on the shuffled data, for each fold and each repeat.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy data synthetic --output toy.h5mu
         drevalpy run Ridge --dataset toy.h5mu --split-mode LPO --no-hpo \
             --randomization-mode SVRC --robustness-repeats 1 --output-dir results_stress

      The command ends with the summary of the ordinary runs:

      .. code-block:: text

         Wrote experiment results to results_stress
         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 1
                 Ridge (5 folds): MSE=0.5648, RMSE=0.7509, MAE=0.5846, R^2=0.7878, Pearson=0.8899, Spearman=0.9006, Kendall=0.7262

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import drevalpy
         from drevalpy import ModelSpec

         dataset = drevalpy.synthetic()
         ridge = ModelSpec.from_name("Ridge")
         results = drevalpy.run(
             [ridge],
             dataset,
             "LPO",
             hyperparameter_tuning=False,
             randomization_modes=["SVRC"],
             robustness_repeats=1,
         )
         print(results)

      The ``print`` shows the summary of the ordinary runs:

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 1
                 Ridge (5 folds): MSE=0.5648, RMSE=0.7509, MAE=0.5846, R^2=0.7878, Pearson=0.8899, Spearman=0.9006, Kendall=0.7262

The summary averages only the ordinary runs: the original data in the original pair order. The
stress-test runs are in the saved results, and no command prints them.

The number of runs per model is the number of folds × (1 + robustness repeats) × (1 + randomized
datasets). Here it is 5 × 2 × 2 = 20 runs of ``Ridge``. With tuning on, every one of these runs
tunes on its own, so the cost of the search multiplies by the same factor; see
:doc:`/how-to/tune_hyperparameters`.

Choose the randomization mode
-----------------------------

A model reads **views** of the dataset: the sample views such as ``gene_expression``, and the drug
views it uses. The mode decides which views are shuffled in each randomized dataset. Repeat
``--randomization-mode`` to run several modes.

.. list-table::
   :header-rows: 1
   :widths: 16 20 64

   * - Mode
     - Views
     - Randomized dataset
   * - ``SVRC``
     - Sample views
     - One per view, with only that view randomized
   * - ``SVCC``
     - Sample views
     - One per view, with all the other views randomized
   * - ``SVRD``
     - Drug views
     - One per drug view, with only that view randomized
   * - ``SVCD``
     - Drug views
     - One per drug view, with all the other drug views randomized

Use ``SVRC`` to ask whether the model needs a view, and ``SVCC`` to ask what a view achieves on
its own. A test that would change nothing is skipped with a warning and produces no run: ``SVCC``
and ``SVCD`` on a model with a single view, which leaves no other view to shuffle, and ``SVRD``
and ``SVCD`` on a dataset that does not store the drug view. A model without views, such as
``NaiveMeanEffectsPredictor``, gets no randomized datasets either. :doc:`/reference/split_modes`
has the exact rules.

``--randomization-type`` chooses how a view is shuffled. ``permutation`` (the default) permutes the
rows across samples or drugs. ``invariant`` replaces each row with a draw from a Gaussian with the
mean and standard deviation of that row, so the shuffled view keeps the scale of each row but
loses the values. ``--randomization-random-state`` fixes the shuffle; it
defaults to ``--seed``.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy run Ridge --dataset toy.h5mu --split-mode LPO --no-hpo \
             --randomization-mode SVRC --randomization-mode SVCC \
             --randomization-type invariant --output-dir results_invariant

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         invariant = drevalpy.run(
             [ridge],
             dataset,
             "LPO",
             hyperparameter_tuning=False,
             randomization_modes=["SVRC", "SVCC"],
             randomization_type="invariant",
         )
         print([(run.randomization, run.robustness_repeat) for run in invariant.models[0].runs][:3])

      Each run records the randomization and the robustness repeat it belongs to:

      .. code-block:: text

         [(None, None), (('SVRC', 'gene_expression'), None), (('SVCC', 'gene_expression'), None)]

Read the stress-test runs
-------------------------

Select runs from a ``ModelResult`` with ``select_runs``. Without arguments it returns the
ordinary runs. ``randomization=(mode, view)`` selects the runs on one randomized dataset and
``robustness=i`` selects repeat ``i``. ``"all"`` for either argument selects every run. Here is a
comparison of the Pearson correlation over all pairs and within each drug:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      Load the saved results directory in Python:

      .. code-block:: python

         import numpy as np
         import pandas as pd

         from drevalpy.types.results import ExperimentResult


         def pearson_scores(runs):
             per_drug = []
             for run in runs:
                 frame = pd.DataFrame({"drug": run.drug_ids, "truth": run.ground_truth, "pred": run.predictions})
                 per_drug.append(frame.groupby("drug").apply(lambda g: g["truth"].corr(g["pred"]), include_groups=False).mean())
             pooled = np.mean([run.metrics["Pearson"] for run in runs])
             return {"all pairs": pooled, "within drug": np.mean(per_drug)}


         model = ExperimentResult.load("results_stress").models[0]
         settings = {
             "ordinary": model.select_runs(),
             "gene_expression permuted": model.select_runs(randomization=("SVRC", "gene_expression")),
             "robustness repeat 0": model.select_runs(robustness=0),
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


         model = results.models[0]
         settings = {
             "ordinary": model.select_runs(),
             "gene_expression permuted": model.select_runs(randomization=("SVRC", "gene_expression")),
             "robustness repeat 0": model.select_runs(robustness=0),
         }
         print(pd.DataFrame({name: pearson_scores(runs) for name, runs in settings.items()}).T.round(3))

Both print:

.. code-block:: text

                              all pairs  within drug
   ordinary                       0.890        0.508
   gene_expression permuted       0.846        0.159
   robustness repeat 0            0.890        0.508

Permuting ``gene_expression`` removes most of the within-drug score, from 0.51 to 0.16, but the
score over all pairs falls much less, because the drug carries most of it. The repeat with
shuffled pair order scores the same as the ordinary run, so the order of the pairs makes no
difference to this model. A model with random components, such as a random forest, can show a spread.
:doc:`/tutorials/tune_and_stress_test` explains how to interpret both results.

Run the tests as separate jobs
------------------------------

``run`` does everything in one process. To repeat the folds with shuffled pair order as separate
jobs, write the shuffled folds with ``drevalpy experiments robustness`` and train each with
``drevalpy single``. To train the randomized datasets as separate jobs, write them with
``drevalpy experiments randomization`` (or ``drevalpy.randomization`` in Python) and train each
with ``single``. ``experiments randomization`` accepts a registered dataset name or a file;
``single`` reads only a file. :doc:`/how-to/run_folds_in_parallel` shows the steps.
