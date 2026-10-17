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
   * - ``--seed``
     - ``42``
     - Root seed for the splits, feature randomization, HPO sampler, and the
       global RNGs before every fit (see :ref:`cli_reproducibility`).
   * - ``--hpo-random-state``
     - unset
     - Optuna sampler seed; defaults to ``--seed``.
   * - ``--randomization-mode`` / ``-r``
     - None
     - Randomization mode(s): ``SVRC``, ``SVCC``, ``SVRD``, ``SVCD``.
   * - ``--randomization-type``
     - ``permutation``
     - ``permutation`` or ``invariant``.
   * - ``--randomization-random-state``
     - unset
     - Root seed for feature randomization; defaults to ``--seed``.
   * - ``--robustness-trials``
     - ``0``
     - Number of robustness permutations (0 = disabled).

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

Options are the same HPO flags (``--hpo``, ``--hpo-metric``,
``--hpo-num-samples``, ``--hpo-random-state``) and ``--seed`` as
``drevalpy run``, plus the following that are specific to per-fold execution:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``--response-transformation``
     - ``standard``
     - Target scaling fitted on the training scope only: ``None``, ``standard``,
       ``minmax``, or ``robust``. Predictions are inverse-transformed before
       scoring, so metrics stay in the dataset's original response units.
   * - ``--split-mode``
     - ``LPO``
     - Split mode recorded when the split file carries none.

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
space are tuned with `Optuna <https://optuna.org/>`_ before final fold
evaluation. Search is Optuna-only: a ``TPESampler`` seeded with
``--hpo-random-state`` (``--seed`` when unset) runs ``--hpo-num-samples`` trials
sequentially in the process and optimizes ``--hpo-metric``. Use ``--no-hpo`` for
defaults-only runs, and run folds as separate ``drevalpy single`` processes to
parallelize.

.. _cli_reproducibility:

Reproducibility
~~~~~~~~~~~~~~~

``--seed`` (default ``42``) controls every random source of a run. ``drevalpy run``
splits with it and uses it as the default for ``--hpo-random-state`` and
``--randomization-random-state``. Both ``run`` and ``single`` seed the Python,
NumPy and torch generators right before each HPO search and final fit, from a
value derived from ``--seed`` and the fold's identity (model, dataset, split
mode, fold, randomization and robustness trial). Re-running with the same
``--seed`` reproduces seeded CPU fits; another ``--seed`` changes them. The seed
is stored in each result's ``fold_metadata["execution"]["seed"]``.

When running folds with ``drevalpy single``, pass the same ``--seed`` to every
fold; the fold's identity keeps their RNG streams distinct.

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
