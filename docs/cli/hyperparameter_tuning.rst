Hyperparameter tuning
=====================

Hyperparameter tuning is **on by default**. When you run the root
``drevalpy`` command, models with a search space are tuned with Ray Tune and
Optuna before final fold evaluation.

Ray vs Optuna
-------------

These are not two alternate backends. They play different roles in one stack:

- **Ray Tune** runs and schedules trials (parallelism, ``--hpo_cpu`` /
  ``--hpo_gpu``, trial storage under the run directory).
- **Optuna** (via Ray’s ``OptunaSearch``) chooses which hyperparameter values
  to try next and optimizes ``--optim_metric``.

Without Ray installed, experiment-time tuning cannot run. Use
``--no_hyperparameter_tuning`` for defaults-only runs, or install on a
platform that has Ray wheels (see :doc:`/getting_started/installation`).

Main flags
----------

- ``--hpo_num_samples`` — number of Optuna trials per fold when tuning is on
  (default ``16``)
- ``--hpo_random_state`` — seed for the Optuna sampler (default ``42``)
- ``--hpo_cpu`` / ``--hpo_gpu`` — Ray resources per trial (optional; GPU
  defaults apply when CUDA is available)
- ``--optim_metric`` — metric Optuna optimizes (default ``RMSE``)
- ``--no_hyperparameter_tuning`` — skip search and use each model’s default
  hyperparameters

Example
-------

Tune baselines on TOYv1 under LPO and keep the search reproducible:

.. code-block:: bash

   drevalpy \
       --run_id tune_baselines \
       --models RandomForest \
       --baselines ElasticNet NaiveMeanEffectsPredictor \
       --dataset_name TOYv1 \
       --test_mode LPO \
       --optim_metric RMSE \
       --hpo_num_samples 16 \
       --hpo_random_state 42

Disable tuning when you only want defaults:

.. code-block:: bash

   drevalpy \
       --run_id defaults_only \
       --models ElasticNet \
       --baselines NaiveMeanEffectsPredictor \
       --dataset_name TOYv1 \
       --no_hyperparameter_tuning

For Weights & Biases logging of trials and metrics, see :doc:`wandb`. For the
full option list, see :doc:`reference`.

Backward compatibility
----------------------

YAML grids
~~~~~~~~~~

Before 1.6.0, baseline tuning often relied on YAML hyperparameter grids.
Those grids are gone for experiment-time search. Each tunable component now
owns a default configuration and a structured Ray/Optuna search space.
``make-hpam-yamls`` no longer expands Cartesian grids — it only writes
``hpam_0.yaml`` with defaults for nf-core wiring. See
:doc:`pipeline_commands`.

``--multiprocessing``
~~~~~~~~~~~~~~~~~~~~~

``--multiprocessing`` remains as a deprecated alias. It only warns; it does
not set trial counts or resources. Use ``--hpo_num_samples`` (and related HPO
flags) instead.
