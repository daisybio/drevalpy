Tune hyperparameters
====================

.. interface: both

Models with hyperparameters are tuned inside every fold, on a validation part of the training
pairs, before the final model is fit. This page shows how to change the budget and the metric of
that search, narrow the search space, set the values yourself, and read what the search chose.
:doc:`/tutorials/tune_and_stress_test` walks through reading the trials of a tuned run.

The examples tune ``ElasticNet`` on the simulated dataset. Keep tuning on for this preset: on
this dataset it scores better tuned than with its default hyperparameters, as
:doc:`/tutorials/first_benchmark` shows.

Set the number of trials
------------------------

A **trial** is one configuration drawn from the search space and scored on the validation pairs.
Tuning is on by default, with 16 trials per fold. Every trial is a model fit, so the number of trials
sets the cost of the experiment: 3 trials on 2 folds fit the model 2 × 3 times for tuning, plus
one final fit per fold. Use a small number to try something out and the default or more for a
result you report. The examples on this page use 2 folds instead of the default 5
(``--n-splits 2``, ``n_splits=2`` in Python) to keep every run short; this gives a rough estimate,
not a result to report.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy data synthetic --output toy.h5mu
         drevalpy experiment ElasticNet --dataset toy.h5mu --split-mode LPO --n-splits 2 --hpo-trials 3

      The command logs one line per trial and fold, and ends with:

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 1
                 ElasticNet (2 folds): MSE=0.6861, RMSE=0.8283, MAE=0.6534, R^2=0.7442, Pearson=0.8684, Spearman=0.8771, Kendall=0.6888

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import drevalpy
         from drevalpy import ModelSpec

         dataset = drevalpy.synthetic()
         elastic_net = ModelSpec.from_name("ElasticNet")
         results = drevalpy.experiment([elastic_net], dataset, "LPO", n_splits=2, hpo_num_trials=3)
         print(results)

      The call logs one line per trial and fold, and the ``print`` shows:

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 1
                 ElasticNet (2 folds): MSE=0.6861, RMSE=0.8283, MAE=0.6534, R^2=0.7442, Pearson=0.8684, Spearman=0.8771, Kendall=0.6888

``--robustness-repeats`` and ``--randomization-mode`` add final fits but no tuning: the runs of
a stress test reuse the best hyperparameters of the original run on the same fold; see
:doc:`/how-to/run_stress_tests`.

Choose the tuning metric
------------------------

The search minimizes ``RMSE`` by default. ``--hpo-metric`` (``hpo_metric`` in Python) takes any
metric DrEvalPy computes: ``MSE``, ``RMSE`` and ``MAE`` are minimized, and ``R^2``, ``Pearson``,
``Spearman`` and ``Kendall`` are maximized. See :doc:`/reference/metrics`. Choose the metric that matches the
question of the experiment, for example ``Pearson`` if you care about ranking and not about the
scale of the response.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy experiment ElasticNet --dataset toy.h5mu --split-mode LPO --n-splits 2 --hpo-trials 3 \
             --hpo-metric Pearson --output-dir results_pearson

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         by_pearson = drevalpy.experiment(
             [elastic_net], dataset, "LPO", n_splits=2, hpo_num_trials=3, hpo_metric="Pearson"
         )

The metric only guides the search. The report and the summary always list all seven metrics.

Narrow the search space
-----------------------

Each hyperparameter of a component has a search space: its type, range and default. The
built-in spaces are wide. When you know a good region, narrow the space: the search then spends
its trials where they count. A ``hyperparameter_space`` on a component **replaces** the built-in
space of that component, so list every parameter of that component you still want to tune. The
entry types, keys and the exact rules are in :doc:`/reference/hyperparameters`.

This example restricts the penalty strength ``alpha`` of the predictor to 0.1 to 10 on a
logarithmic scale.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      A recipe carries no hyperparameter space, so describe the model in a YAML file in an
      extension directory, here ``my_models/NarrowElasticNet.yaml``. The file name is the zoo name.

      .. code-block:: yaml

         sample_featurizer: scaledGeneExpression
         drug_featurizer: fingerprints
         predictor:
           name: elasticNet
           hyperparameter_space:
             alpha: {type: float, low: 0.1, high: 10.0, log: true, default: 1.0}
             l1_ratio: {type: float, low: 0.0, high: 1.0, default: 0.5}

      Load the directory with the global ``-e`` option and use the name like any other:

      .. snippet: skip

      .. code-block:: bash

         drevalpy -e my_models experiment NarrowElasticNet \
             --dataset toy.h5mu --split-mode LPO --n-splits 2 --hpo-trials 3

      See :doc:`/how-to/extend/load_extensions` for extension directories.

   .. tab-item:: Python
      :sync: python

      Build a ``ModelConfig`` from a dictionary and give it a name with ``ModelSpec``:

      .. code-block:: python

         from drevalpy.models.config import from_dict

         space = {
             "alpha": {"type": "float", "low": 0.1, "high": 10.0, "log": True, "default": 1.0},
             "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0, "default": 0.5},
         }
         config = from_dict(
             {
                 "sample_featurizer": "scaledGeneExpression",
                 "drug_featurizer": "fingerprints",
                 "predictor": {"name": "elasticNet", "hyperparameter_space": space},
             }
         )
         narrow = ModelSpec("NarrowElasticNet", config)
         narrowed = drevalpy.experiment([narrow], dataset, "LPO", n_splits=2, hpo_num_trials=3)
         print(narrowed)

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 1
                 NarrowElasticNet (2 folds): MSE=0.6165, RMSE=0.7850, MAE=0.6181, R^2=0.7709, Pearson=0.8857, Spearman=0.8956, Kendall=0.7159

The qualified key of a hyperparameter in a composed model, such as ``predictor.elasticNet.alpha``,
is also in ``narrow.hyperparameter_space``.

Set the values yourself
-----------------------

To skip the search, fit with fixed values. With tuning off, every fold uses the **defaults** of
the model. Set a default under ``defaults`` in the mapping form of the predictor, here
``alpha`` = 0.1. This does not change the range of the space, so it matters when tuning is
off. For a single fit with values of your choice, without any experiment, see
:doc:`/how-to/train_save_load`.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      Put this in ``my_models/FixedElasticNet.yaml``:

      .. code-block:: yaml

         sample_featurizer: scaledGeneExpression
         drug_featurizer: fingerprints
         predictor:
           name: elasticNet
           defaults:
             alpha: 0.1

      Then run it with tuning off:

      .. snippet: skip

      .. code-block:: bash

         drevalpy -e my_models experiment FixedElasticNet \
             --dataset toy.h5mu --split-mode LPO --n-splits 2 --no-hpo

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         fixed_config = from_dict(
             {
                 "sample_featurizer": "scaledGeneExpression",
                 "drug_featurizer": "fingerprints",
                 "predictor": {"name": "elasticNet", "defaults": {"alpha": 0.1}},
             }
         )
         fixed = ModelSpec("FixedElasticNet", fixed_config)
         fixed_results = drevalpy.experiment([fixed], dataset, "LPO", n_splits=2, hyperparameter_tuning=False)
         print(fixed_results)

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 1
                 FixedElasticNet (2 folds): MSE=0.7052, RMSE=0.8398, MAE=0.6617, R^2=0.7371, Pearson=0.8650, Spearman=0.8736, Kendall=0.6847

Reproduce a search
------------------

The sampler of the search is seeded with ``--seed``, the root seed of the experiment, so the same
command repeats the same trials. ``--hpo-random-state`` (``hpo_random_state``) sets the seed of
the sampler alone, to vary the search while the folds stay the same.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy experiment ElasticNet --dataset toy.h5mu --split-mode LPO --n-splits 2 --hpo-trials 3 \
             --hpo-random-state 7 --output-dir results_hpo_seed7

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         other_search = drevalpy.experiment([elastic_net], dataset, "LPO", n_splits=2, hpo_num_trials=3, hpo_random_state=7)

Read what the search chose
--------------------------

Every run stores the hyperparameters of its final model and the trials that led to them. They
are kept in the saved results, so you can read them after the experiment. Look at fold 0 of the
tuned ``ElasticNet``:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command line prints no trials. Load the results directory in Python with
      ``ExperimentResult.load("results")`` and read the runs as in the Python tab.
      :doc:`/tutorials/tune_and_stress_test` shows the code.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         run = results.models[0].select_runs()[0]
         print(len(run.trials), "trials, scored by", run.trials[0].optimization_metric)
         print({key: round(run.best_hyperparameters[f"predictor.elasticNet.{key}"], 4) for key in ("alpha", "l1_ratio")})

      .. code-block:: text

         3 trials, scored by RMSE
         {'alpha': 0.0029, 'l1_ratio': 0.3664}

``best_hyperparameters`` holds every hyperparameter of the model under its qualified key, such as
``predictor.elasticNet.alpha``; see :doc:`/reference/hyperparameters`.

Each fold tunes on its own validation pairs, so different folds can choose different values. A
run without tuning has no trials. A big spread between folds, or a best value at the edge of the
range, suggests that the range is too narrow or the dataset too small to tell the configurations
apart.
