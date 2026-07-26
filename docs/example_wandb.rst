DrEvalPy and Weights & Biases
======================================

Enabling wandb
--------------

Pass ``--wandb_project`` to enable wandb logging.

.. code-block:: bash

    drevalpy \
        --run_id my_wandb_run \
        --models RandomForest \
        --baselines ElasticNet NaiveMeanEffectsPredictor \
        --dataset_name TOYv1 \
        --path_data data \
        --wandb_project my_new_project_name

Each model run logs config such as ``model_name``, ``dataset``, ``test_mode``,
``split_index``, and the hyperparameters used for that run. After final
training on the train+validation fold, test metrics are logged with a
``test_`` prefix — always ``test_R^2`` and ``test_Pearson``, plus
``test_<optim_metric>`` (for example ``test_RMSE`` when
``--optim_metric RMSE``). Some neural models also stream epoch-level train/val
metrics during fitting.

Authentication
--------------

Auth follows wandb's usual mechanisms:

1. The ``WANDB_API_KEY`` environment variable
2. Stored credentials from a previous ``wandb login``
3. Interactive login

Wandb with hyperparameter tuning
--------------------------------

Hyperparameter tuning is **on by default** (Ray Tune + Optuna). Use
``--hpo_num_samples`` for the number of trials and ``--optim_metric`` for the
metric Optuna optimizes. Pass ``--no_hyperparameter_tuning`` only if you want
each model's default hyperparameters without search.

Example: tune and compare baselines while logging to wandb:

.. code-block:: bash

    drevalpy \
        --run_id compare_baselines \
        --models RandomForest \
        --baselines ElasticNet GradientBoosting AdaBoostDecisionTree NaiveMeanEffectsPredictor \
        --dataset_name TOYv1 \
        --path_data data \
        --test_mode LPO \
        --n_cv_splits 5 \
        --optim_metric Pearson \
        --hpo_num_samples 16 \
        --hpo_random_state 42 \
        --wandb_project compare_baselines

This tunes each model for up to 16 trials per fold (when the model has a search
space), logs trials and metrics to wandb, and compares the baselines under the
same split settings.

Inspecting hyperparameter effects
---------------------------------

In the wandb UI, use ``+ Add Panels`` to add visualizations. Add
``Parameter Importance`` (for example with respect to ``val_R^2``) and select
the hyperparameters you care about:

.. image:: _static/img/wandb_parameter_importance.png
   :alt: Parameter importance displayed by wandb
   :align: center
   :width: 100%

Add a ``Parallel Coordinates Plot`` as well:

.. image:: _static/img/wandb_parallel_coords.png
   :alt: Parallel coordinates plot
   :align: center
   :width: 100%

Filter to dig deeper — for example ``split_index=4`` and
``model_name="Elastic Net"``:

.. image:: _static/img/wandb_parallel_coords2.png
   :alt: Parallel coordinates plot Elastic Net
   :align: center
   :width: 100%
