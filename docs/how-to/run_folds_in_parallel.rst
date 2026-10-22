Run folds in parallel and combine them
======================================

.. interface: both

``run`` trains every model on every fold in one process. To spread the work over several
processes, machines or cluster jobs, split the dataset once, train each model on each fold as a
separate job, and combine the finished runs into one experiment. The combined experiment gives the
same numbers as ``run`` with the same seed, and every later step (reports, loading, metrics)
works on it as on any other. This page shows the steps. :doc:`/how-to/run_experiment` covers the
single-process route and :doc:`/reference/results` the files the steps write.

The examples use the simulated dataset and ``NaiveMeanEffectsPredictor``, which runs offline in
seconds. For a real model, repeat the training step once per model and fold.

Split the dataset into folds
----------------------------

Write the folds to disk once, before any job starts, so that every job reads the same split.
The fold files are small and hold only masks over the pairs of the dataset; see
:doc:`/how-to/load_and_split`.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy data synthetic --output toy.h5mu
         drevalpy data split toy.h5mu splits --mode LPO

      The command writes ``splits/fold_0.npz`` to ``splits/fold_4.npz``.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import drevalpy
         from drevalpy import ModelSpec

         dataset = drevalpy.synthetic()
         folds = drevalpy.split(dataset, "LPO")
         print(len(folds))

      .. code-block:: text

         5

Train one model on one fold
---------------------------

One ``single`` call trains one model on one fold and returns one run. Its options match the
ones of ``run`` that apply to a single fit: ``--hpo-trials``, ``--hpo-metric``,
``--response-transformation`` and ``--seed``; see :doc:`/how-to/tune_hyperparameters`. Keep the
default seed unless you also change it in the comparison runs, because the seed and the fold
decide the result.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command takes the model, the dataset file, a fold file and the file to write. Each of
      the five calls is independent and can run on its own machine:

      .. code-block:: bash

         drevalpy single NaiveMeanEffectsPredictor toy.h5mu splits/fold_0.npz runs/NaiveMeanEffectsPredictor_0.npz --no-hpo
         drevalpy single NaiveMeanEffectsPredictor toy.h5mu splits/fold_1.npz runs/NaiveMeanEffectsPredictor_1.npz --no-hpo
         drevalpy single NaiveMeanEffectsPredictor toy.h5mu splits/fold_2.npz runs/NaiveMeanEffectsPredictor_2.npz --no-hpo
         drevalpy single NaiveMeanEffectsPredictor toy.h5mu splits/fold_3.npz runs/NaiveMeanEffectsPredictor_3.npz --no-hpo
         drevalpy single NaiveMeanEffectsPredictor toy.h5mu splits/fold_4.npz runs/NaiveMeanEffectsPredictor_4.npz --no-hpo

      Each call ends with the path of the run it wrote:

      .. code-block:: text

         Result: NaiveMeanEffectsPredictor on SYNTHETIC (fold 0) -> runs/NaiveMeanEffectsPredictor_0.npz

      On a machine with a shell, a loop does the same. A job scheduler runs the body of the loop
      once per task instead:

      .. snippet: skip
      .. code-block:: bash

         for i in 0 1 2 3 4; do
             drevalpy single NaiveMeanEffectsPredictor toy.h5mu splits/fold_$i.npz runs/NaiveMeanEffectsPredictor_$i.npz --no-hpo &
         done
         wait

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         spec = ModelSpec.from_name("NaiveMeanEffectsPredictor")
         runs = [drevalpy.single(spec, dataset, fold, hyperparameter_tuning=False) for fold in folds]
         print(len(runs))

      .. code-block:: text

         5

      ``single`` returns a ``RunResult``. To train in several processes, make each worker load
      the dataset and the fold itself, as in this script. The ``if __name__ == "__main__"`` guard
      is required:

      .. snippet: skip
      .. code-block:: python

         from concurrent.futures import ProcessPoolExecutor

         import drevalpy
         from drevalpy import ModelSpec
         from drevalpy.types import SplitMasks
         from drevalpy.types.results import ExperimentResult


         def train(fold_path):
             dataset = drevalpy.load("toy.h5mu")
             fold = SplitMasks.load(fold_path)
             spec = ModelSpec.from_name("NaiveMeanEffectsPredictor")
             return drevalpy.single(spec, dataset, fold, hyperparameter_tuning=False)


         if __name__ == "__main__":
             paths = [f"splits/fold_{i}.npz" for i in range(5)]
             with ProcessPoolExecutor(max_workers=2) as pool:
                 runs = list(pool.map(train, paths))
             ExperimentResult(runs).save("agg")

Combine the runs
----------------

Combine the runs of all models and folds into one experiment. ``aggregate`` takes every run file
explicitly, because a shell glob is not expanded in every context where you run commands. It
groups the runs by model and sorts the folds, so the order of the files does not matter.
``--group-by`` also prints the metrics computed within each drug (``drug``) or sample
(``sample``) instead of pooled over all pairs.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy aggregate \
             runs/NaiveMeanEffectsPredictor_0.npz runs/NaiveMeanEffectsPredictor_1.npz \
             runs/NaiveMeanEffectsPredictor_2.npz runs/NaiveMeanEffectsPredictor_3.npz \
             runs/NaiveMeanEffectsPredictor_4.npz \
             --output-dir agg --group-by drug

      The command writes the experiment to ``agg`` and prints its summary:

      .. code-block:: text

         Aggregated 5 runs into ExperimentResult at agg
         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 1
                 NaiveMeanEffectsPredictor (5 folds): MSE=0.8399, RMSE=0.9144, MAE=0.7218, R^2=0.6851, Pearson=0.8298, Spearman=0.8409, Kendall=0.6466
         Metrics computed within each drug, averaged over drugs and folds (number of drugs scored in brackets):
         Model                      MSE        RMSE       MAE        R^2         Pearson    Spearman   Kendall
         NaiveMeanEffectsPredictor  0.852 (8)  0.895 (8)  0.731 (8)  -0.583 (8)  0.386 (8)  0.372 (8)  0.277 (8)

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy.types.results import ExperimentResult

         experiment = ExperimentResult(runs)
         experiment.save("agg")
         print(experiment)

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 1
                 NaiveMeanEffectsPredictor (5 folds): MSE=0.8399, RMSE=0.9144, MAE=0.7218, R^2=0.6851, Pearson=0.8298, Spearman=0.8409, Kendall=0.6466

These are the numbers that ``drevalpy run NaiveMeanEffectsPredictor --dataset toy.h5mu --split-mode LPO --no-hpo``
prints for the same dataset and seed. The experiment directory loads with
``ExperimentResult.load`` and feeds ``drevalpy report`` as any other; see
:doc:`/how-to/make_report`. Compare several models by passing the runs of all of them to one
``aggregate`` call.

Run the robustness test as separate jobs
----------------------------------------

A robustness test repeats each fold with the order of the pairs shuffled; see
:doc:`/how-to/run_stress_tests`. ``drevalpy experiments robustness`` writes the shuffled folds as
fold files named ``fold_<i>_trial_<j>.npz``. They go through the same ``single`` call. The runs
join the others in ``aggregate``, which saves them with the experiment, and the summary still
averages only the ordinary runs.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy experiments robustness splits rob --n-permutations 1
         drevalpy single NaiveMeanEffectsPredictor toy.h5mu rob/fold_0_trial_0.npz runs/rob_0.npz --no-hpo

      Train the other folds the same way, then pass all the run files to ``aggregate``. The
      folds ``rob/fold_<i>_trial_0.npz`` give the same runs as the first repeat of
      ``--robustness-trials`` in ``run``.

   .. tab-item:: Python
      :sync: python

      The folds of a repeat come from the fold files only, so in Python use the command above on
      the fold files you wrote, or run the whole test in one process with ``drevalpy.run(...,
      robustness_trials=1)``, as in :doc:`/how-to/run_stress_tests`.

Run the randomization test as separate jobs
-------------------------------------------

The randomization test trains the model on datasets whose views are shuffled, so the unit of
work is a shuffled dataset and a fold. In Python, ``drevalpy.randomization`` returns the shuffled
datasets for a model and a mode, and ``single`` trains on each. The run records the randomization
it belongs to, and the record survives ``save`` and ``load``. On the command line, run the test
inside ``drevalpy run`` with ``-r``: the shuffled datasets that ``drevalpy experiments
randomization`` writes cannot be trained with ``drevalpy single`` yet.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      Use ``drevalpy run --randomization-mode SVRC`` to run the test; see
      :doc:`/how-to/run_stress_tests`.

   .. tab-item:: Python
      :sync: python

      ``NaiveMeanEffectsPredictor`` reads no views, so there is nothing to shuffle and the list
      is empty. ``Ridge`` reads ``gene_expression``, so ``SVRC`` gives one shuffled dataset:

      .. code-block:: python

         print(len(drevalpy.randomization(spec, dataset, ["SVRC"])))
         ridge = ModelSpec.from_name("Ridge")
         shuffled = drevalpy.randomization(ridge, dataset, ["SVRC"])
         print(len(shuffled))
         stress_run = drevalpy.single(ridge, shuffled[0], folds[0], hyperparameter_tuning=False)
         print(stress_run.randomization)

      .. code-block:: text

         0
         1
         ('SVRC', 'gene_expression')

      Add the shuffled run to the list of runs before ``ExperimentResult(runs)``, and train it
      for every fold. A saved experiment keeps the randomization of each run.
