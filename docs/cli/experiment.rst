Run an experiment
=================

The root ``drevalpy`` command runs the full drug-response evaluation pipeline:
load data, split, tune (by default), train, test, and write predictions under
``results/``.

Minimal example:

.. code-block:: bash

   drevalpy \
       --run_id my_run \
       --models ElasticNet \
       --baselines NaiveMeanEffectsPredictor \
       --dataset_name TOYv1 \
       --test_mode LCO

This page covers the important options in prose. For the full generated
inventory, see :doc:`reference`. For datasets, split modes, metrics, and the
model zoo, see :doc:`/concepts/datasets` and :doc:`/concepts/evaluation`.

Identity and paths
------------------

``--run_id`` names the result tree (default ``my_run``). ``--path_data`` is
where datasets and features are stored or downloaded (default ``data``).
``--path_out`` is the results root (default ``results/``). With the defaults,
outputs appear under ``results/<run_id>/<dataset_name>/<test_mode>/``.

Use ``--overwrite`` if you need to replace an existing run with the same
``--run_id`` and ``--path_out``.

Models and baselines
--------------------

``--models`` lists models to evaluate (including randomization and robustness
when you enable those). ``--baselines`` lists comparison models; they are tuned
and scored but skip randomization and robustness. If you omit
``NaiveMeanEffectsPredictor`` from baselines, DrEvalPy adds it — evaluation
depends on it.

Pass several names space-separated, for example
``--models RandomForest ElasticNet``.

Dataset and response
--------------------

``--dataset_name`` selects a built-in dataset (default ``GDSC1``) or a custom
name under ``--path_data``. ``--measure`` chooses the response column (default
``LN_IC50``). Unless you pass ``--no_refitting``, DrEvalPy uses CurveCurator
refit measures (for example ``LN_IC50_curvecurator``) for better
cross-dataset comparability.

For custom raw viability, place ``<dataset_name>_raw.csv`` under
``--path_data`` and leave refitting enabled. ``--curve_curator_cores`` and
``--curve_curator_normalize`` control fitting. See
:doc:`/concepts/datasets`.

``--cross_study_datasets`` adds external datasets for cross-study prediction
checks after the main run.

Split settings
--------------

``--test_mode`` selects one or more leave-out protocols: ``LPO``, ``LCO``,
``LTO``, ``LDO`` (default ``LPO``). ``--n_cv_splits`` defaults to ``7`` —
you need at least seven folds for a meaningful critical-difference diagram.

For external split scripts, see :doc:`custom_splits`.

Hyperparameter tuning
---------------------

Tuning is on by default. ``--optim_metric`` chooses the metric used for
hyperparameter optimization; the default is ``RMSE``. Other common choices
include ``MSE``, ``MAE``, ``R^2``, ``Pearson``, ``Spearman``, and ``Kendall``.

Control search with ``--hpo_num_samples`` (default ``16``),
``--hpo_random_state``, and optional ``--hpo_cpu`` / ``--hpo_gpu`` resources
per trial. Pass ``--no_hyperparameter_tuning`` to skip search and use each
model’s default hyperparameters.

Details, including the Ray/Optuna backend: :doc:`hyperparameter_tuning`.

Randomization and robustness
----------------------------

``--randomization_mode`` adds feature-shuffle tests (``SVCC``, ``SVRC``,
``SVCD``, ``SVRD``; ``None`` disables). ``--randomization_type`` is
``permutation`` (default) or ``invariant``. ``--n_trials_robustness`` repeats
training with different seeds; ``0`` (default) disables the robustness test.

These extras apply to ``--models`` only, not baselines. See
:doc:`/concepts/evaluation`.

Logging and checkpoints
-----------------------

``--wandb_project`` enables Weights & Biases logging for every model in the
run — see :doc:`wandb`. ``--model_checkpoint_dir`` stores neural checkpoints
(default ``TEMPORARY``). ``--final_model_on_full_data`` trains and tunes a
final model on the union of all folds after CV.

``--response_transformation`` optionally scales the target during training
(``standard``, ``minmax``, or ``robust``) and inverts it for final
predictions.

Nextflow for large runs
-----------------------

For demanding or highly reproducible workloads, use
`nf-core/drugresponseeval <https://nf-co.re/drugresponseeval/dev/>`_. The
pipeline calls the stepwise ``drevalpy`` subcommands documented in
:doc:`pipeline_commands`.

After a local run, generate the HTML report with :doc:`reporting`.
