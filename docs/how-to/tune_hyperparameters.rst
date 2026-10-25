Tune hyperparameters
====================

.. interface: both

Models with hyperparameters are tuned inside every fold, on a validation part of the training
pairs, before the final model is fit. This page shows how to change the budget and the metric of
that search, narrow the search space, set the values yourself, and read what the search chose.
:doc:`/tutorials/tune_and_stress_test` walks through reading the trials of a tuned run.

The examples tune ``ElasticNet`` on the simulated dataset. Keep tuning on for this preset: with
its default hyperparameters it predicts a constant on this dataset.

Set the number of trials
------------------------

A **trial** is one configuration drawn from the search space and scored on the validation pairs.
Tuning is on by default, with 16 trials per fold. Every trial is a model fit, so the number of trials
sets the cost of the experiment: 4 trials on 5 folds fit the model 5 × 4 times for tuning, plus
one final fit per fold. Use a small number to try something out and the default or more for a
result you report.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy data synthetic --output toy.h5mu
         drevalpy run ElasticNet --dataset toy.h5mu --split-mode LPO --hpo-trials 4

      The command logs one line per trial and fold, and ends with:

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 1
                 ElasticNet (5 folds): MSE=0.5495, RMSE=0.7407, MAE=0.5832, R^2=0.7939, Pearson=0.8935, Spearman=0.9024, Kendall=0.7275

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import drevalpy
         from drevalpy import ModelSpec

         dataset = drevalpy.synthetic()
         elastic_net = ModelSpec.from_name("ElasticNet")
         results = drevalpy.run([elastic_net], dataset, "LPO", hpo_num_trials=4)
         print(results)

      The call logs one line per trial and fold, and the ``print`` shows:

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 1
                 ElasticNet (5 folds): MSE=0.5495, RMSE=0.7407, MAE=0.5832, R^2=0.7939, Pearson=0.8935, Spearman=0.9024, Kendall=0.7275

Every run of a stress test tunes on its own, so ``--robustness-repeats`` and
``--randomization-mode`` multiply the cost of tuning too; see :doc:`/how-to/run_stress_tests`.

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

         drevalpy run ElasticNet --dataset toy.h5mu --split-mode LPO --hpo-trials 4 \
             --hpo-metric Pearson --output-dir results_pearson

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         by_pearson = drevalpy.run([elastic_net], dataset, "LPO", hpo_num_trials=4, hpo_metric="Pearson")

The metric only guides the search. The report and the summary always list all seven metrics.

Narrow the search space
-----------------------

Each hyperparameter of a component has a search space: its type, range and default. The
built-in spaces are wide. When you know a good region, narrow the space: the search then spends
its trials where they count. A ``hyperparameter_space`` on a component **replaces** the built-in
space of that component, so list every parameter of that component you still want to tune. The
entry types, keys and the exact rules are in :doc:`/reference/hyperparameters`.

This example restricts the penalty strength ``alpha`` of the predictor to 0.01 to 1 on a
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
             alpha: {type: float, low: 0.01, high: 1.0, log: true, default: 0.1}
             l1_ratio: {type: float, low: 0.0, high: 1.0, default: 0.5}

      Load the directory with the global ``-e`` option and use the name like any other:

      .. snippet: skip

      .. code-block:: bash

         drevalpy -e my_models run NarrowElasticNet \
             --dataset toy.h5mu --split-mode LPO --hpo-trials 4

      See :doc:`/how-to/extend/load_extensions` for extension directories.

   .. tab-item:: Python
      :sync: python

      Build a ``ModelConfig`` from a dictionary and give it a name with ``ModelSpec``:

      .. code-block:: python

         from drevalpy.models.config import from_dict

         space = {
             "alpha": {"type": "float", "low": 0.01, "high": 1.0, "log": True, "default": 0.1},
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
         narrowed = drevalpy.run([narrow], dataset, "LPO", hpo_num_trials=4)
         print(narrowed)

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 1
                 NarrowElasticNet (5 folds): MSE=0.5464, RMSE=0.7385, MAE=0.5787, R^2=0.7951, Pearson=0.8936, Spearman=0.9021, Kendall=0.7273

The qualified key of a hyperparameter in a composed model, such as ``predictor.elasticNet.alpha``,
is also in ``narrow.hyperparameter_space``.

Set the values yourself
-----------------------

To skip the search, fit with fixed values. With tuning off, every fold uses the **defaults** of
the model. Set a default by giving the parameter as a value in the one-key form of the predictor,
here ``alpha`` = 0.01. This does not change the range of the space, so it matters when tuning is
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
           elasticNet:
             alpha: 0.01

      Then run it with tuning off:

      .. snippet: skip

      .. code-block:: bash

         drevalpy -e my_models run FixedElasticNet \
             --dataset toy.h5mu --split-mode LPO --no-hpo

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         fixed_config = from_dict(
             {
                 "sample_featurizer": "scaledGeneExpression",
                 "drug_featurizer": "fingerprints",
                 "predictor": {"elasticNet": {"alpha": 0.01}},
             }
         )
         fixed = ModelSpec("FixedElasticNet", fixed_config)
         fixed_results = drevalpy.run([fixed], dataset, "LPO", hyperparameter_tuning=False)
         print(fixed_results)

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LPO
             Normalized by: None
             Models: 1
                 FixedElasticNet (5 folds): MSE=0.5552, RMSE=0.7445, MAE=0.5839, R^2=0.7915, Pearson=0.8917, Spearman=0.9017, Kendall=0.7270

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

         drevalpy run ElasticNet --dataset toy.h5mu --split-mode LPO --hpo-trials 4 \
             --hpo-random-state 7 --output-dir results_hpo_seed7

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         other_search = drevalpy.run([elastic_net], dataset, "LPO", hpo_num_trials=4, hpo_random_state=7)

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
         print({key: round(run.best_hyperparameters[key], 4) for key in ("alpha", "l1_ratio")})

      .. code-block:: text

         4 trials, scored by RMSE
         {'alpha': 0.1091, 'l1_ratio': 0.1705}

Each fold tunes on its own validation pairs, so different folds can choose different values. A
run without tuning has no trials. A big spread between folds, or a best value at the edge of the
range, suggests that the range is too narrow or the dataset too small to tell the configurations
apart.
