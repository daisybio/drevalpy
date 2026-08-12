Experiments
===========

If you are reading this, we assume you are already familiar with these
concepts:

- :doc:`/concepts/evaluation`
- :doc:`/concepts/from_components_to_models`

The DrEvalPy CLI provides commands for the full experiment lifecycle: running
a complete pipeline, executing individual folds, and combining results.

``drevalpy run`` — full pipeline
---------------------------------

The ``run`` command loads a dataset, splits it, tunes models (by default),
trains, predicts, and writes results:

.. code-block:: bash

   drevalpy run ElasticNet RandomForest \
       --dataset GDSC1 \
       --split-mode LCO \
       --output-dir results

Models are passed as positional arguments. Options:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``--dataset`` / ``-d``
     -
     - Dataset name or ``.h5mu`` path (required).
   * - ``--split-mode`` / ``-s``
     - ``LPO``
     - Split mode: ``LPO``, ``LCO``, ``LDO``, or ``LTO``.
   * - ``--output-dir`` / ``-o``
     - ``results``
     - Output directory for results.
   * - ``--hpo`` / ``--no-hpo``
     - ``--hpo``
     - Enable or disable hyperparameter tuning.
   * - ``--hpo-metric``
     - ``RMSE``
     - Metric to optimize (``MSE``, ``MAE``, ``R^2``, ``Pearson``, ``Spearman``, ``Kendall``).
   * - ``--hpo-num-samples``
     - ``16``
     - Number of Optuna trials per fold.
   * - ``--hpo-random-state``
     - ``42``
     - HPO random seed.
   * - ``--randomization-mode`` / ``-r``
     - None
     - Randomization mode(s): ``SVRC``, ``SVCC``, ``SVRD``, ``SVCD``.
   * - ``--randomization-type``
     - ``permutation``
     - ``permutation`` or ``invariant``.
   * - ``--robustness-trials``
     - ``0``
     - Number of robustness permutations (0 = disabled).
   * - ``--precomputed-only``
     - off
     - Restrict HPO to pre-computed featurizer variants.

Example with tuning disabled:

.. code-block:: bash

   drevalpy run ElasticNet --dataset GDSC1 --split-mode LCO --no-hpo

Example with custom HPO settings:

.. code-block:: bash

   drevalpy run RandomForest \
       --dataset GDSC1 \
       --split-mode LPO \
       --hpo-metric Pearson \
       --hpo-num-samples 32 \
       --hpo-random-state 123

``drevalpy single`` — per-fold execution
-----------------------------------------

For parallel or distributed workflows, run individual folds separately:

.. code-block:: bash

   drevalpy single ElasticNet data/GDSC1.h5mu splits/fold_0.npz results/fold_0.npz \
       --hpo-metric RMSE \
       --hpo-num-samples 16

Arguments:

1. Model name (zoo preset or custom)
2. Dataset path (``.h5mu`` file)
3. Split path (``.npz`` fold file from ``drevalpy data split``)
4. Output path (``.npz`` result file)

Options are the same HPO flags as ``drevalpy run``.

``drevalpy aggregate`` — combine results
-----------------------------------------

Combine per-fold ``RunResult`` files into a single ``ExperimentResult``:

.. code-block:: bash

   drevalpy aggregate results/fold_0.npz results/fold_1.npz results/fold_2.npz \
       --output-dir experiment_results

The aggregated result can then be passed to ``drevalpy report``.

Hyperparameter tuning
---------------------

Tuning is **on by default** (``--hpo``). When enabled, models with a search
space are tuned with Ray Tune and Optuna before final fold evaluation.

Ray vs Optuna
~~~~~~~~~~~~~

These are not two alternate backends. They play different roles in one stack:

- **Ray Tune** runs and schedules trials (parallelism, resource allocation,
  trial storage under the run directory).
- **Optuna** (via Ray's ``OptunaSearch``) chooses which hyperparameter values
  to try next and optimizes ``--hpo-metric``.

Without Ray installed, experiment-time tuning cannot run. Use ``--no-hpo``
for defaults-only runs, or install on a platform that has Ray wheels (see
:doc:`/getting_started/installation`).

Randomization and robustness
-----------------------------

``--randomization-mode`` adds feature-shuffle tests (``SVCC``, ``SVRC``,
``SVCD``, ``SVRD``). ``--randomization-type`` is ``permutation`` (default)
or ``invariant``. ``--robustness-trials`` repeats training with shuffled fold
orderings; ``0`` disables.

For standalone randomization and robustness:

.. code-block:: bash

   drevalpy experiments randomization ElasticNet GDSC1 randomized/ --mode SVRC
   drevalpy experiments robustness splits/ robustness_splits/ --n-permutations 5

Weights & Biases
----------------

Hyperparameter search can log each trial to `Weights & Biases
<https://wandb.ai/>`_, but only from Python: ``hpam_tune`` accepts a
``wandb_project`` argument, and no CLI command forwards it. Call the tuning
API directly if you need W&B logging.

Nextflow for large runs
-----------------------

For demanding or highly reproducible workloads, use
`nf-core/drugresponseeval <https://nf-co.re/drugresponseeval/dev/>`_. The
pipeline pins the ``drevalpy`` version it runs and is driven by its own
`pipeline parameters <https://nf-co.re/drugresponseeval/dev/parameters/>`_,
which are distinct from the CLI options above.
