Migrate from DrEvalPy 1.x
=========================

DrEvalPy v2 is a rewrite, not an upgrade in place. The three guides below cover
the three ways people used 1.x. Each lists every 1.x option, function or class
it knows about and says what replaces it, or that nothing does.

.. toctree::
   :maxdepth: 1

   cli
   python
   models

- :doc:`cli` is for people who ran ``drevalpy`` or the ``drevalpy-*`` scripts
  from a shell, a workflow engine or a job script.
- :doc:`python` is for people who imported ``drevalpy`` in a script or
  notebook: ``MODEL_FACTORY``, ``drug_response_experiment``,
  ``DrugResponseDataset``.
- :doc:`models` is for people who wrote a ``DRPModel`` subclass and registered
  it in ``MODEL_FACTORY``.

What changed
------------

- The word "cell line" became "sample" in every name, because datasets can hold
  organoids, PDX models and primary patient cells. The mapping is in
  :ref:`how-to/migrate/index:Renamed terms`.
- A dataset is one ``.h5mu`` file with a samples-by-drugs response matrix and
  the feature tables. 1.x kept a long response table and loose feature CSVs in a
  data directory.
- A model is a pair of components, a featurizer and a predictor, named by a zoo
  preset or a recipe. 1.x models were subclasses of ``DRPModel``.
- The command line has the subcommands ``run``, ``single``, ``aggregate``,
  ``curate``, ``report``, ``data``, ``experiments`` and ``list``. Bare
  ``drevalpy`` prints the help and does not run an experiment.
- Hyperparameter tuning is an Optuna search of ``--hpo-trials`` trials per
  fold. 1.x tried every combination of a grid in ``hyperparameters.yaml``.

.. _migrate-files:

Which 1.x files still load
--------------------------

None of them. Re-run the experiment and retrain the model. This is what v2
checks when you try:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 1.x file
     - What v2 reads instead
     - What happens when you load the 1.x file
   * - Predictions and metrics: ``predictions_split_N.csv``,
       ``evaluation_results*.csv``, ``true_vs_pred.csv``
     - ``.npz`` files that ``run``, ``single`` and ``aggregate`` write
       (``RunResult``, ``ExperimentResult``).
     - ``RunResult.load`` opens the file with NumPy and refuses pickled data,
       so a CSV raises ``ValueError``. v2 has no CSV importer.
   * - Splits: ``cv_split_N_train.csv``, ``..._validation.csv``,
       ``..._test.csv``
     - One ``fold_N.npz`` per fold from ``drevalpy data split``: boolean masks
       over the samples-by-drugs response matrix.
     - Not readable. Create new splits, or write a splitter for your own fold
       logic (:doc:`/how-to/extend/write_splitter`).
   * - Model directory written by a 1.x ``DRPModel.save(directory)``
     - A single ``.zip`` archive of format ``drevalpy-model``.
     - ``DRPModel.load`` raises ``ValueError``: the checkpoint path "must be an
       archive file path, not a directory".
   * - A model checkpoint of another format version
     - Format version 1 (``FORMAT_VERSION`` in
       ``drevalpy/models/mixins/_persistence_io.py``).
     - ``UnsupportedCheckpointFormatError`` naming the stored version and the
       one this release reads; retrain the model and save it again.

Do not combine old and new output in one comparison. A model with the same name
in 1.x and v2 is a reimplementation built from components, so its numbers can
differ; see :doc:`/explanation/literature_models` for the literature models.

.. _migrate-renames:

Renamed terms
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 1.x
     - v2
   * - ``cell_line_ids`` (``DrugResponseDataset``, result files)
     - ``sample_ids`` (``Dataset.sample_ids``, ``RunResult.sample_ids``)
   * - ``cell_line_views``, ``cell_line_input``,
       ``load_cell_line_features``
     - ``sample_views``; the sample side of a featurizer. See :doc:`models`.
   * - ``--test_mode``, ``test_mode=``
     - ``--split-mode``, ``split_mode=``
   * - ``LCO`` (leave-cell-line-out)
     - ``LSO`` (leave-sample-out). ``LCO`` is still accepted as an alias and is
       always recorded as ``LSO``. See :doc:`/reference/split_modes`.
   * - ``--optim_metric``, ``hpam_optimization_metric=``
     - ``--hpo-metric``, ``hpo_metric=``
   * - ``--n_trials_robustness``, ``n_trials_robustness=``
     - ``--robustness-trials``, ``robustness_trials=``
   * - ``NaiveCellLineMeanPredictor``
     - ``NaiveSampleMeanPredictor``
   * - ``cell_line_name`` and ``cellosaurus_id`` as required dataset columns
     - Optional columns, present only in datasets whose samples are cell lines.
       A dataset needs a unique sample ID as the observation index, and a
       ``tissue`` column. See :doc:`/reference/datasets`.

The randomization modes keep their codes (``SVCC``, ``SVRC``, ``SVCD``,
``SVRD``). The ``C`` codes act on the sample side.

.. _migrate-models:

Map 1.x model names to v2
-------------------------

The first column is the name you passed to ``--models`` or looked up in
``MODEL_FACTORY``. The second is the name to use in v2. Every name in the second
column is a zoo preset; check yours with ``drevalpy list zoo-presets``.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - 1.x name (``MODEL_FACTORY`` key)
     - v2 zoo name
     - Note
   * - ``NaivePredictor``
     - ``NaivePredictor``
     -
   * - ``NaiveCellLineMeanPredictor``
     - ``NaiveSampleMeanPredictor``
     - Renamed.
   * - ``NaiveDrugMeanPredictor``
     - ``NaiveDrugMeanPredictor``
     -
   * - ``NaiveTissueMeanPredictor``
     - ``NaiveTissueMeanPredictor``
     -
   * - ``NaiveTissueDrugMeanPredictor``
     - ``NaiveTissueDrugMeanPredictor``
     -
   * - ``NaiveMeanEffectsPredictor``
     - ``NaiveMeanEffectsPredictor``
     - v2 never adds it for you. Name it in the model list if you want to
       normalize against it.
   * - ``ElasticNet`` (class ``ElasticNetModel``)
     - ``ElasticNet``
     - Only the class name changed in 1.x terms; the key is the same.
   * - ``Lasso`` (class ``LassoModel``)
     - ``Lasso``
     -
   * - ``SVR`` (class ``SVMRegressor``)
     - ``SVR``
     -
   * - ``RandomForest``
     - ``RandomForest``
     -
   * - ``GradientBoosting``
     - ``GradientBoosting``
     -
   * - ``KNNRegressor``
     - ``KNNRegressor``
     -
   * - ``AdaBoostDecisionTree``
     - ``AdaBoostDecisionTree``
     -
   * - ``MultiViewRandomForest``
     - ``MultiViewRandomForest``
     -
   * - ``MultiViewXGBoost``
     - ``MultiViewXGBoost``
     -
   * - ``MultiViewLightGBM``
     - ``MultiViewLightGBM``
     -
   * - ``SimpleNeuralNetwork``
     - ``SimpleNeuralNetwork``
     -
   * - ``MultiViewNeuralNetwork``
     - ``MultiViewNeuralNetwork``
     -
   * - ``SingleDrugElasticNet``
     - ``SingleDrugElasticNet``
     - Single-drug model. One run fits every drug; there is no per-drug job to
       consolidate.
   * - ``SingleDrugRandomForest``
     - ``SingleDrugRandomForest``
     - Single-drug model, as above.
   * - ``DrugGNN``
     - ``DrugGNN``
     -
   * - ``DIPK`` (class ``DIPKModel``)
     - ``DIPK``
     -
   * - ``PharmaFormer`` (class ``PharmaFormerModel``)
     - ``PharmaFormer``
     -
   * - ``Precily`` (class ``PrecilyModel``)
     - ``Precily``
     -
   * - ``SparseGO`` (class ``SparseGOModel``)
     - ``SparseGO``
     -
   * - ``SRMF``
     - ``SRMF``
     -
   * - ``MOLIR``
     - ``MOLIR``
     - Single-drug model. Needs the ``copy_number_variation_gistic`` view.
   * - ``SuperFELTR``
     - ``SuperFELTR``
     - Single-drug model.
   * - ``EnsembleMF``
     - Removed
     - No zoo preset and no registered component. Nothing replaces it.
   * - ``PaccMann``
     - Removed
     - No zoo preset and no registered component. Nothing replaces it.

v2 adds ``Ridge``, which 1.x did not have. ``drevalpy list zoo-presets`` shows
what each preset is made of; :doc:`/reference/model_zoo` describes them.

Where to go next
----------------

- :doc:`/tutorials/first_benchmark` runs a first benchmark on a synthetic
  dataset, with no download.
- :doc:`/how-to/run_experiment` is the v2 version of what
  ``drevalpy --models ...`` did.
- :doc:`/reference/glossary` defines the v2 terms used in these guides.
