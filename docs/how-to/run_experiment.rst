Run an experiment
=================

.. interface: both

An experiment evaluates one or more models on one dataset over the folds of one split mode. This
page shows how to run it, change what it does, and read the results back. :doc:`/tutorials/first_benchmark`
walks through a first run and what its numbers mean; this page is the reference-style recipe for
repeating it on your own data.

The examples use the simulated dataset and naive baselines, so they run offline in seconds.

Run models on a dataset
-----------------------

Give ``run`` the models, a dataset and a split mode. The models are zoo names (see
:doc:`/how-to/choose_models`). The dataset is a registered name or the path to an ``.h5mu`` file
(see :doc:`/how-to/load_and_split`). The split mode is ``LPO``, ``LSO``, ``LDO`` or ``LTO``; the
default on the command line is ``LPO``. ``run`` always splits into 5 folds and trains every model
on every fold.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy data synthetic --output toy.h5mu
         drevalpy run NaiveDrugMeanPredictor NaiveMeanEffectsPredictor \
             --dataset toy.h5mu --split-mode LPO --no-hpo --output-dir results_lpo

      The command writes the results to ``results_lpo`` and ends with the summary of the
      experiment:

      .. code-block:: text

         Wrote experiment results to results_lpo
         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 2
                 NaiveDrugMeanPredictor (5 folds): MSE=0.7908, RMSE=0.8880, MAE=0.7195, R^2=0.7031, Pearson=0.8415, Spearman=0.8440, Kendall=0.6882
                 NaiveMeanEffectsPredictor (5 folds): MSE=0.8399, RMSE=0.9144, MAE=0.7218, R^2=0.6851, Pearson=0.8298, Spearman=0.8409, Kendall=0.6466

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import drevalpy
         from drevalpy import ModelSpec

         dataset = drevalpy.synthetic()
         models = [ModelSpec.from_name(name) for name in ["NaiveDrugMeanPredictor", "NaiveMeanEffectsPredictor"]]
         results = drevalpy.run(models, dataset, "LPO", hyperparameter_tuning=False)
         print(results)

      ``run`` returns an ``ExperimentResult`` and does not write anything to disk. The
      ``print`` shows the same summary as the command:

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 2
                 NaiveDrugMeanPredictor (5 folds): MSE=0.7908, RMSE=0.8880, MAE=0.7195, R^2=0.7031, Pearson=0.8415, Spearman=0.8440, Kendall=0.6882
                 NaiveMeanEffectsPredictor (5 folds): MSE=0.8399, RMSE=0.9144, MAE=0.7218, R^2=0.6851, Pearson=0.8298, Spearman=0.8409, Kendall=0.6466

Each metric in the summary is computed once per fold on the pooled test pairs and averaged over
the folds. Read it together with the within-drug scores; see :doc:`/explanation/metrics`.

``--no-hpo`` and ``hyperparameter_tuning=False`` switch tuning off, which suits the naive
baselines because they have no hyperparameters. Leave tuning on for models that have
them, and see :doc:`/how-to/tune_hyperparameters`.

Use a built-in dataset
----------------------

Pass the registered name instead of a path. The first run downloads the dataset into the cache
directory.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. snippet: network

      .. code-block:: bash

         drevalpy run NaiveDrugMeanPredictor NaiveMeanEffectsPredictor \
             --dataset TOYv1 --split-mode LPO --no-hpo --output-dir results_toyv1

   .. tab-item:: Python
      :sync: python

      .. snippet: network

      .. code-block:: python

         results_toyv1 = drevalpy.run(models, "TOYv1", "LPO", hyperparameter_tuning=False)

``run`` also accepts a path string in place of a name or a ``Dataset``.

Change the split mode
---------------------

The split mode decides what the models are asked to generalize to; see
:doc:`/reference/split_modes` and :doc:`/explanation/evaluation_design`. One run covers one split
mode, so run the experiment once per mode and keep the results apart.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy run NaiveDrugMeanPredictor NaiveMeanEffectsPredictor \
             --dataset toy.h5mu --split-mode LSO --no-hpo --output-dir results_lso

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         by_mode = {
             mode: drevalpy.run(models, dataset, mode, hyperparameter_tuning=False)
             for mode in ["LPO", "LSO"]
         }
         print({mode: result.split_mode for mode, result in by_mode.items()})

      .. code-block:: text

         {'LPO': 'LPO', 'LSO': 'LSO'}

Control the randomness
----------------------

``--seed`` (default 42) is the root seed of the splits, the tuning and every model fit. Runs with
the same seed, data and models use the same folds. Use a different seed to see how much the
scores depend on the particular folds. ``--hpo-random-state`` overrides the seed of the tuning
sampler alone.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy run NaiveDrugMeanPredictor NaiveMeanEffectsPredictor \
             --dataset toy.h5mu --split-mode LPO --no-hpo --seed 7 --output-dir results_seed7

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         reseeded = drevalpy.run(models, dataset, "LPO", hyperparameter_tuning=False, seed=7)

Scale the response
------------------

``--response-transformation`` scales the response before the models train, with ``standard``,
``minmax`` or ``robust`` scaling. The default is ``none``. The scaler is fitted on the training
pairs of each fold, and the predictions are transformed back, so the scores stay in the units of
the response. In Python, pass an unfitted scikit-learn transformer to ``response_transformation``;
DrEvalPy clones and fits it for every fit.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy run NaiveMeanEffectsPredictor \
             --dataset toy.h5mu --split-mode LPO --no-hpo --response-transformation standard \
             --output-dir results_scaled

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from sklearn.preprocessing import StandardScaler

         scaled = drevalpy.run(
             models[1:], dataset, "LPO", hyperparameter_tuning=False, response_transformation=StandardScaler()
         )

Read the results
----------------

The command line saves the results for you. In Python, save them yourself with
``results.save(directory)``. Both write the layout in :doc:`/reference/results`: one
subdirectory per model with one ``.npz`` file per run. Load a directory back into an
``ExperimentResult`` to recompute tables or build a report, even in a later session:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      ``drevalpy report`` and ``drevalpy aggregate`` read the directories directly; see
      :doc:`/how-to/make_report`. Add ``--group-by drug`` or ``--group-by sample`` to ``run`` to
      print the metrics computed within each drug or sample as well.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy.types.results import ExperimentResult

         results.save("results_lpo")
         loaded = ExperimentResult.load("results_lpo")
         print(loaded.model_names)
         print(round(loaded.summary_table()["NaiveDrugMeanPredictor"]["Pearson"], 3))

      .. code-block:: text

         ['NaiveDrugMeanPredictor', 'NaiveMeanEffectsPredictor']
         0.842

      ``summary_table()`` returns the pooled metrics of each model. ``summary_table(group_by="drug")``
      computes them within each drug, as in the tutorial.

Where to go next
----------------

- :doc:`/how-to/tune_hyperparameters` for models that have hyperparameters.
- :doc:`/how-to/run_stress_tests` to test whether a model uses its features.
- :doc:`/how-to/run_folds_in_parallel` to train the folds as separate jobs.
- :doc:`/how-to/make_report` for the HTML report.
