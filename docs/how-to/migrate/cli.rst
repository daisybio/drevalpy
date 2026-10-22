Migrate command-line runs from 1.x
==================================

.. interface: cli

In 1.x, ``drevalpy`` with flags ran a whole experiment, and 17 ``drevalpy-*``
scripts ran its steps one by one for the Nextflow pipeline. In v2 the
``drevalpy`` command only has subcommands, and it is the only script. This page
maps every 1.x flag and script to its v2 command.

Run an experiment
-----------------

The 1.x call below ran two models and the baselines on GDSC1, in
leave-cell-line-out mode, without tuning.

.. snippet: skip

.. code-block:: bash

   drevalpy --run_id my_first_run --models ElasticNet RandomForest --baselines NaiveMeanEffectsPredictor --dataset_name GDSC1 --test_mode LCO --no_hyperparameter_tuning --path_out results

The v2 call names the dataset file, one split mode and the output directory. The
baselines are ordinary models in the same list. This version makes a synthetic
dataset first, so it runs without a download:

.. code-block:: bash

   drevalpy data synthetic --output toy.h5mu
   drevalpy run NaiveTissueMeanPredictor NaiveDrugMeanPredictor NaiveMeanEffectsPredictor --dataset toy.h5mu --split-mode LSO --no-hpo --output-dir results/my_first_run

``--split-mode LCO`` is accepted as an alias of ``LSO``. To cover several split
modes, as the 1.x ``--test_mode LPO LCO`` did, call ``run`` once per mode with
its own output directory:

.. code-block:: bash

   drevalpy run NaiveTissueMeanPredictor NaiveDrugMeanPredictor NaiveMeanEffectsPredictor --dataset toy.h5mu --split-mode LPO --no-hpo --output-dir results/my_first_run_lpo

Then write the report from one results directory. ``--reference-model`` plays
the part of the 1.x ``NaiveMeanEffectsPredictor`` normalization:

.. code-block:: bash

   drevalpy report results/my_first_run --reference-model NaiveMeanEffectsPredictor --output-dir report

What changed in how a run behaves:

- ``run`` always makes five folds. 1.x made ``--n_cv_splits`` folds, seven by
  default.
- ``run`` does not check whether the output directory already holds results, so
  there is no ``--overwrite``. Use a new directory for each run.
- v2 does not add ``NaiveMeanEffectsPredictor`` for you, and the response
  transformation applies to every model you list.
- ``--no-hpo`` trains each model with its default hyperparameters. 1.x used the
  first entry of ``hyperparameters.yaml``.
- With tuning on (``--hpo``, the default), each fold gets an Optuna search of
  ``--hpo-trials`` trials (16 by default), not a full grid.

Map the ``drevalpy`` flags
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 1.x flag
     - v2
   * - ``--run_id``
     - Removed. Name the run with ``--output-dir``.
   * - ``--path_data``
     - Removed. Pass the dataset file with ``--dataset``. Datasets downloaded
       by name are cached under ``DREVALPY_CACHE_DIR``
       (:doc:`/reference/environment`).
   * - ``--models``
     - The positional ``MODELS`` arguments of ``run``. Names are in
       :ref:`how-to/migrate/index:Map 1.x model names to v2`.
   * - ``--baselines``
     - Put the baseline in the same ``MODELS`` list.
   * - ``--test_mode``
     - ``--split-mode`` (``-s``). One mode per call, default ``LPO``. See
       :doc:`/reference/split_modes`.
   * - ``--randomization_mode``
     - ``--randomization-mode`` (``-r``). Repeat the flag for several modes:
       ``-r SVCC -r SVRC``. A space-separated list is read as model names.
   * - ``--randomization_type``
     - ``--randomization-type``. The default is unchanged.
   * - ``--n_trials_robustness``
     - ``--robustness-trials``. The default is ``0``.
   * - ``--dataset_name``
     - ``--dataset`` (``-d``), now required. It takes a ``.h5mu`` path, or the
       name of a built-in dataset: ``BeatAML2``, ``CTRPv1``, ``CTRPv2``,
       ``GDSC1``, ``GDSC2``, ``PDX_Bruna`` or ``TOYv1``. See
       ``drevalpy list datasets``. There is no default dataset.
   * - ``--cross_study_datasets``
     - Removed. Nothing replaces it.
   * - ``--path_out``
     - ``--output-dir`` (``-o``), default ``results``.
   * - ``--no_refitting``
     - Removed. Nothing replaces it.
   * - ``--curve_curator_cores``, ``--curve_curator_normalize``
     - ``drevalpy curate --cores`` and ``--normalize``. See
       `Fit your own raw data`_.
   * - ``--measure``
     - Removed. A v2 dataset stores one response matrix, in pEC50. Fit other
       measures into it with ``drevalpy curate``.
   * - ``--overwrite``
     - Removed. See above.
   * - ``--optim_metric``
     - ``--hpo-metric``, default ``RMSE``.
   * - ``--wandb_project``
     - Removed. Nothing replaces it.
   * - ``--n_cv_splits``
     - Removed for ``run``, which always makes five folds.
       ``drevalpy data split --n-splits`` sets the count for splits you make
       yourself.
   * - ``--response_transformation``
     - ``--response-transformation`` with ``none`` (default), ``standard``,
       ``minmax`` or ``robust``. The 1.x values ``drug_mean`` and
       ``drug_tissue_mean`` are gone.
   * - ``--multiprocessing``
     - Removed (it started Ray). Run folds in parallel with ``single``; see
       :doc:`/how-to/run_folds_in_parallel`.
   * - ``--model_checkpoint_dir``
     - Removed from the command line. Checkpoints are zip archives written
       with ``DRPModel.save``; see :doc:`/how-to/train_save_load`.
   * - ``--final_model_on_full_data``
     - Removed. See `Replace the pipeline scripts`_.
   * - ``--no_hyperparameter_tuning``
     - ``--no-hpo``. The default is ``--hpo``.
   * - ``--custom_splitter_path``, ``--custom_split_name``
     - Removed. Register a splitter in an extension file and load it with
       ``drevalpy -e DIR``. See :doc:`/how-to/extend/write_splitter`.
   * - ``--clean_min_responders``, ``--clean_min_responder_frac``
     - Removed. The splitters drop pairs that fail the default curve-quality
       thresholds. That is a filter on pairs, not on drugs.
   * - ``--version``, ``-v``
     - Removed. Read ``drevalpy.__version__`` in Python.

New options that 1.x did not have: ``--seed`` (default ``42``),
``--hpo-random-state`` and ``--randomization-random-state``.

Replace the pipeline scripts
----------------------------

Each 1.x script was a deprecated alias of a subcommand with the same name
without ``drevalpy-``. v2 has none of them, and the pipeline they served no
longer has a counterpart: v2 does not split a run into cross-validation,
hyperparameter, randomization and final-model jobs. The table shows which v2
command covers the job.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 1.x script
     - v2
   * - ``drevalpy-report``
     - ``drevalpy report EXPERIMENT_DIR``. It takes the directory ``run`` or
       ``aggregate`` wrote, not ``--run_id``, ``--dataset_name`` and
       ``--path_data``. See `Read the results`_.
   * - ``drevalpy-viability-preprocess``, ``drevalpy-viability-postprocess``
     - ``drevalpy curate``. See `Fit your own raw data`_.
   * - ``drevalpy-load-response``
     - ``drevalpy data load`` reads a dataset. There is no cross-study dataset
       and no responder filter.
   * - ``drevalpy-make-cv-pkls``
     - ``drevalpy data split DATASET OUTPUT_DIR --mode MODE``. It writes
       ``fold_N.npz`` files, not pickles.
   * - ``drevalpy-make-hpam-yamls``, ``drevalpy-evaluate-hpams``
     - Removed. Tuning runs inside ``run`` and ``single``; see
       :doc:`/how-to/tune_hyperparameters`.
   * - ``drevalpy-train-cv``, ``drevalpy-test-cv``
     - ``drevalpy single MODEL DATASET SPLIT OUTPUT``, which trains, tunes and
       scores one model on one fold.
   * - ``drevalpy-make-randomization-yamls``
     - ``drevalpy experiments randomization MODEL DATASET OUTPUT_DIR`` writes
       the randomized datasets.
   * - ``drevalpy-make-final-split-pkls``, ``drevalpy-tune-final-model``,
       ``drevalpy-train-final-model``
     - Removed. There is no final production model on the full data. To keep a
       trained model, save it from Python; see :doc:`/how-to/train_save_load`.
   * - ``drevalpy-consolidate-single-drug``
     - Removed. A single-drug model is one model in v2 and needs no
       consolidation.
   * - ``drevalpy-evaluate-test``
     - Removed. ``run`` and ``single`` score the test fold themselves.
   * - ``drevalpy-collect-results``
     - ``drevalpy aggregate RESULTS... --output-dir experiment_results``.
   * - ``drevalpy-make-pipeline-report``
     - ``drevalpy report``.

Split a run into jobs
---------------------

To spread a run over a cluster, as the scripts did, split the dataset once, run
``single`` for each fold and model, and gather the outputs with ``aggregate``.
This example uses three folds of leave-sample-out on the synthetic dataset:

.. code-block:: bash

   drevalpy data split toy.h5mu splits --mode LSO --n-splits 3
   drevalpy single NaiveTissueMeanPredictor toy.h5mu splits/fold_0.npz out/f0.npz --no-hpo
   drevalpy single NaiveTissueMeanPredictor toy.h5mu splits/fold_1.npz out/f1.npz --no-hpo
   drevalpy single NaiveTissueMeanPredictor toy.h5mu splits/fold_2.npz out/f2.npz --no-hpo
   drevalpy aggregate out/f0.npz out/f1.npz out/f2.npz --output-dir experiment_results

``single`` has the same ``--hpo``, ``--hpo-metric``, ``--hpo-trials``,
``--seed`` and ``--response-transformation`` options as ``run``. It has no
option for a hyperparameter file; ``--no-hpo`` uses the preset's defaults.
:doc:`/how-to/run_folds_in_parallel` shows how to hand these commands to a
scheduler. The stress tests have their own commands, covered in
:doc:`/how-to/run_stress_tests`.

Fit your own raw data
---------------------

The 1.x raw-data scripts read a CSV with the columns ``dose``, ``response``,
``sample``, ``drug`` and an optional ``replicate``, with doses in µM. ``drevalpy
curate`` reads a table with the columns ``drug``, ``sample``, ``concentration``
and ``intensity``, and an optional ``replicate``, again in µM. It fits a
dose-response curve per pair and writes an ``.h5ad`` file. ``run`` needs an
``.h5mu`` file with the feature tables added, as
:doc:`/how-to/bring_your_own_data` describes.

The 1.x "prefit" CSV with the columns ``cell_line_name``, ``pubchem_id`` and a
measure has no direct reader. Rebuild it as a dataset by that guide.

Read the results
----------------

The 1.x layout was ``results/<run_id>/<dataset>/<mode>/<model>/predictions/``
and a report ``index.html`` next to four ``evaluation_results*.csv`` and
``true_vs_pred.csv`` files. In v2:

- ``run`` writes ``metadata.json`` and, for each model, a directory with
  ``metadata.json`` and one ``fold_N.npz`` per fold.
- ``report`` writes ``<title>_multiqc_report.html`` and, next to it, a
  ``..._multiqc_report_data`` directory. Its
  ``dreval_summary_table.txt`` and ``dreval_summary_table_per_drug.txt`` replace
  ``evaluation_results.csv`` and ``evaluation_results_per_drug.csv``.
  ``evaluation_results_per_cl.csv`` and ``true_vs_pred.csv`` have no file
  replacement. Load the results in Python to get them; see
  :doc:`python`.
- The report is MultiQC, which is installed with DrEvalPy. No extra is needed.
- ``report`` has ``--title`` (``-t``) and ``--reference-model`` (``-r``). It accepts ``--dataset``
  for compatibility and ignores it.

:doc:`/how-to/make_report` covers the report in full.

Old files
---------

No 1.x output loads in v2: not the prediction CSVs, not the split files and not
the model directories. Re-run the experiment. The table of what v2 does when you
try is at :ref:`how-to/migrate/index:Which 1.x files still load`.
