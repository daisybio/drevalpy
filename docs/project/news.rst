News
====

.. role:: date
    :class: date

🧬 DrEvalPy 2.0.0 :date:`2026-10-22`
--------------------------------------------------------------
v2 is a rewrite of the interface, not an upgrade in place. It is not released yet:
``pip install drevalpy`` still installs 1.6.0. :doc:`/how-to/migrate/index` maps every 1.x
option, function and model to its v2 counterpart. This entry lists what changed for users.

Breaking changes from 1.6.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Nothing that 1.x wrote loads in v2.** Re-run experiments and retrain models. Result files,
  splits and model checkpoints are in new formats; the checkpoint format version restarts at 1.
  :ref:`how-to/migrate/index:Which 1.x files still load` lists what each loader says.
- **The command line is one script.** ``drevalpy`` has the subcommands ``run``, ``single``,
  ``aggregate``, ``curate``, ``report``, ``data``, ``experiments`` and ``list``, and the 17
  ``drevalpy-*`` scripts are removed. Options are renamed; see :doc:`/how-to/migrate/cli`.
- **The report takes no dataset.** ``drevalpy report --dataset`` is removed, because the report never
  read the file. A pipeline that still passes it, such as nf-core/drugresponseeval, must drop the flag.
- **The Python API is new.** ``MODEL_FACTORY``, ``drug_response_experiment`` and
  ``DrugResponseDataset`` are replaced by ``ModelSpec``, ``drevalpy.run`` and ``Dataset``; see
  :doc:`/how-to/migrate/python`. A 1.x ``DRPModel`` subclass becomes a featurizer and a predictor;
  see :doc:`/how-to/migrate/models`.
- **Importing the package no longer configures logging.** Call ``drevalpy.log.setup_logging()`` to
  see progress and warnings from a script or notebook. The command line does it for you.
- **"Cell line" is "sample"** in every name, as described next.
- **Tuning is a serial Optuna search**, not a grid. Ray and the ``multiprocessing`` extra are
  gone; run folds in parallel with ``drevalpy single`` instead
  (:doc:`/how-to/run_folds_in_parallel`).

Samples instead of cell lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Datasets can hold organoids, patient-derived xenografts and primary patient cells, so every name
that said "cell line" now says "sample".

- ``cell_line_ids`` is ``sample_ids``, the ``cell_line_featurizer`` key of a model configuration
  and of zoo files is ``sample_featurizer``, ``cell_line_views`` is ``sample_views``,
  ``drevalpy list cell-line-featurizers`` is ``drevalpy list sample-featurizers``, and
  ``n_cell_lines=`` of the synthetic dataset builder is ``n_samples=``.
  ``NaiveCellLineMeanPredictor`` is ``NaiveSampleMeanPredictor``.
- The split mode leave-cell-line-out (LCO) is leave-sample-out (LSO). ``LCO`` is still accepted
  as an alias, for example in ``--split-mode LCO``, and results always record ``LSO``.
- A dataset needs a unique observation index and a ``tissue`` column. The columns
  ``cellosaurus_id`` and ``cell_line_name`` are optional and only make sense for datasets whose
  samples are cell lines. See :doc:`/reference/datasets`.

Defaults and results that changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``--response-transformation`` defaults to ``none`` in ``drevalpy run`` and ``drevalpy single``,
  so the response is not standardized unless you ask for it. The values are ``none``, ``standard``,
  ``minmax`` and ``robust``, plus the 1.x group means ``drug_mean`` and ``drug_tissue_mean``.
- ``ExperimentResult.summary_table`` is a method, not a property. Call ``summary_table()`` for
  the pooled metrics, or ``summary_table(group_by="drug")``, ``"sample"`` or ``"study"``
  to compute each metric within every drug, sample or study and average over the groups.
  ``group_by`` also accepts a sample or drug annotation column of the dataset, such as
  ``group_by="tissue"`` or ``"ncit_class"`` with ``dataset=`` the dataset of the run. A group
  where the data cannot support the metric (too few pairs, or a constant response for a correlation
  or ``R^2``) is skipped for every model alike. A correlation of a constant prediction within a
  group is 0, not undefined, and counts in the mean, so ``NaiveDrugMeanPredictor`` scores 0 per
  drug. ``summary_group_counts`` returns how many groups each value is based
  on. Both return a ``pandas.DataFrame`` with one row per model and one column per metric.
- ``drevalpy run`` and ``drevalpy aggregate`` take ``--group-by drug|sample|study`` or the name of a
  sample or drug annotation column such as ``tissue`` and print that table. ``run`` reads the annotation
  from its dataset; ``aggregate`` needs ``--dataset``. The report has a second summary table, "Model Summary Table per Drug".
- One ``--seed`` (``seed=`` in Python, default 42) drives the splits, the hyperparameter search,
  the randomization tests and the seeding of every model fit. ``drevalpy data split`` and
  ``drevalpy experiments randomization`` take ``--seed`` as well.
- Leaderboard titles state the real number of folds, for example "5-Fold Leave-Sample-Out".
- The standard deviation of a metric across folds is the sample standard deviation (``ddof=1``), and
  ``0`` for a single fold.
- **Saved results store each value once.** The ``metadata.json`` of an experiment directory holds only
  ``normalized_by`` and ``models``, a model directory has no ``metadata.json``, and the dataset name, the
  split mode and the model name are read from the ``fold_<i>.npz`` files, which load in numeric order. The
  split mode is always stored under ``split_mode``. :doc:`/reference/results` has the layout.
- **Models use the 1.6.x gene lists.** The sklearn models, ``SimpleNeuralNetwork``, ``DrugGNN``,
  ``PharmaFormer``, ``DIPK``, ``EnsembleMF`` and ``SingleDrugKNN`` read the 867 landmark genes, ``PaccMann``
  its network-propagation list, ``RandomForestAEMolGNet`` the TCGA autoencoder list, and the ``MultiView*``
  models, ``MOLIR`` and ``SuperFELTR`` the lists of their views, instead of every gene of the view. Scores
  differ from a model that reads the whole view. The lists ship inside the package as Parquet files and
  :ref:`reference/datasets:Gene lists` describes them.
- **A missing gene is an error.** If a dataset lacks genes of a list, the featurizer raises an
  ``InvalidInputError`` that names the list, the view and the number of missing genes, instead of silently
  keeping a smaller selection. ``gene_list=None`` (``gene_list_stem`` for ``landmarkGenes``) keeps the whole
  view. ``PDX_Bruna`` and ``BeatAML2`` lack genes of several lists; see :doc:`/how-to/bring_your_own_data`.
- **Default hyperparameters follow 1.6.1** for the sklearn, boosted-tree, neural-network, ``PharmaFormer`` and
  ``SparseGO`` presets, and ``GradientBoosting`` is seeded with ``random_state=42``. The values are in
  :doc:`/reference/hyperparameters`.
- **The DIPK autoencoder and the PharmaFormer scalers are fitted on each training sample once**, not once per
  sample-drug pair, as in 1.6.1. The DIPK autoencoder z-scores every sample on its own, trains for 1000 epochs
  by default, and takes ``lr_autoencoder``, ``patience_autoencoder`` and ``batch_size_autoencoder``.
- **One rule for constant values.** The pooled and the per-group scores treat a vector as constant when
  ``max`` and ``min`` are within ``numpy.isclose`` of each other (``atol=1e-6``), whatever the order of the
  values, and apply one rule:

  - an error metric is ``NaN`` with no pair;
  - ``R^2`` and the correlations are ``NaN`` with fewer than two pairs or a constant response;
  - a correlation of a constant prediction with a varying response is ``0.0``;
  - ``R^2`` of a constant prediction is the usual ``1 - SSE/SST``.

  Two pooled results differ from 1.x: ``R^2`` on a constant response is ``NaN`` (scikit-learn gave
  ``0.0`` or ``1.0``), and a correlation with both sides constant is ``NaN`` (1.x gave ``0.0``).
- **One list of metrics.** ``drevalpy.evaluation.Metric`` lists the metrics; each member is its name
  (``Metric.PEARSON == "Pearson"``) and has a ``.direction`` (``"min"`` or ``"max"``) and a ``.kind``. Each
  metric still calls its scipy or scikit-learn function, now for pooled and per-group scores alike.
  ``evaluate(predictions, response, metrics)`` returns pooled scores and
  ``evaluate_per_group(predictions, response, metrics, labels=...)`` per-group scores. ``AVAILABLE_METRICS``,
  ``MINIMIZATION_METRICS``, ``MAXIMIZATION_METRICS``, ``get_mode``, ``pearson``, ``spearman``, ``kendall``,
  ``is_constant``, ``ranges_are_constant`` and ``CONSTANT_ATOL`` are removed, and ``evaluate`` no longer takes a
  dataset. ``--hpo-metric`` lists the metric names and rejects an unknown one before any data is loaded, and
  ``HPOConfig`` checks its metric and derives the Optuna direction from it instead of storing a ``mode``;
  ``drevalpy.models.tuning.validate_hpo_metric`` is removed.
- **The mean-effects baseline works without tissue labels.** ``NaiveMeanEffectsPredictor`` on a dataset with no
  tissue information predicts the dataset mean plus the sample and drug effects.
- **The leaderboard has four panels:** the Pearson correlation (normalized when the report is), the RMSE, the raw
  Pearson correlation and the Pearson correlation within each drug. The normalization reference stays in the
  leaderboard and the critical difference diagram as a baseline row.

New models
~~~~~~~~~~

Four models from the 1.6.x line are zoo presets: ``SingleDrugKNN`` (a nearest-neighbour model per drug with
mRMR feature selection), ``RandomForestAEMolGNet`` (a random forest on DIPK's autoencoder and MolGNet
features), ``EnsembleMF`` and ``PaccMann``. Every 1.6.1 model has a preset of the same name, except
``NaiveCellLineMeanPredictor``, which is ``NaiveSampleMeanPredictor``. The table is at
:ref:`how-to/migrate/index:Map 1.x model names to v2`, and :doc:`/explanation/literature_models` describes
``EnsembleMF`` and ``PaccMann``.

New commands and functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``drevalpy train MODEL --dataset DATASET --output MODEL_DIR`` trains a final model on all pairs of a
  dataset, and ``drevalpy predict MODEL_DIR --dataset DATASET --output FILE`` applies it. In Python they are
  ``drevalpy.train`` and the saved ``DRPModel``. See :doc:`/how-to/train_save_load`.
- ``drevalpy --version`` prints the installed version.
- The response transformations ``drug_mean`` and ``drug_tissue_mean`` subtract the mean of each drug, or of
  each drug and tissue, from the response. ``NaiveMeanEffectsPredictor`` is always fitted on the
  untransformed response.
- ``drevalpy.synthetic()`` and ``drevalpy data synthetic --output toy.h5mu`` build a small
  simulated dataset, by default 60 samples, 8 drugs and 6 tissues with a gene expression view.
  It needs no download, and its drugs differ so much in average potency that pooled scores
  look good for models that rank samples poorly. ``drevalpy.testing.build_synthetic_dataset``
  is an alias.
- ``drevalpy data split`` and ``drevalpy experiments randomization`` accept a registered
  dataset name as well as a file, like ``drevalpy run --dataset``. ``drevalpy single`` reads
  only a file.
- ``drevalpy list zoo-presets`` and ``drevalpy list datasets`` show the model presets you can
  pass to ``run`` and the datasets you can load by name.
- ``DREVALPY_EXTENSIONS_DIR`` is read by ``load_extensions()`` in Python, not only by the
  command line. It can name several directories, separated by ``:`` on Linux and macOS.
- ``register_source`` and ``register_dataset`` raise ``ValueError`` when the name is already
  registered with a different entry, including a built-in one. Pass ``override=True`` to
  replace it. Registering an identical entry again does nothing.

Errors and logging
~~~~~~~~~~~~~~~~~~

- A mistake in a command, such as an unknown model name or a missing file, prints
  ``Error: <message>`` and exits with status 1, without a traceback. An unexpected failure still
  shows its traceback.
- The new module ``drevalpy.errors`` holds the library's errors. ``DrEvalError`` is the base class;
  ``UnknownNameError`` (also a ``ValueError`` and a ``KeyError``), ``InputNotFoundError`` (also a
  ``FileNotFoundError``) and ``InvalidInputError`` (also a ``ValueError``) cover the common cases, so
  existing ``except`` clauses keep working.
- ``drevalpy`` takes the global options ``--verbose`` / ``-v`` (debug logging) and ``--quiet`` /
  ``-q`` (warnings only). By default the report logs one line per plot instead of one per model of
  every per-model plot; ``-v`` brings the per-model timings back.
- The randomization tests write their datasets to files named ``<mode>__<view>.h5mu``.
- ``drevalpy run`` has no short ``-r`` for ``--randomization-mode``; repeat the long flag. A
  mode, ``--fit-speed`` or ``--join`` value that does not exist is rejected with the list of valid
  choices (``drevalpy curate`` no longer takes ``--fit-type``, since OLS was its only value).
  ``drevalpy data synthetic`` prints an error when ``--n-drugs`` is over the number of drugs it
  can simulate.

Cross-study evaluation, component grids and recorded failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Cross-study evaluation.** ``drevalpy.data.concat`` and ``drevalpy data concat`` join the
  datasets of several studies into one dataset. Sample IDs become ``<study>::<sample_id>``,
  ``obs`` gains ``study`` and ``source_sample_id``, and ``join="inner"`` (``--join inner``) keeps
  only the features that all studies share. The new split mode ``LStO`` (leave-study-out) holds
  out one study per fold. ``drevalpy run`` takes several ``--dataset`` options, and
  ``drevalpy.run`` a list of datasets, and concatenates them. ``--group-by study`` and
  ``summary_table(group_by="study")`` score each study separately, and the report adds a per-study
  section to the comparison scatter plot. See :doc:`/how-to/run_cross_study`.
- **Component grids and recipe names.** ``drevalpy run -S/-D/-P`` and
  ``drevalpy.models.combinations`` evaluate every valid combination of sample featurizers, drug
  featurizers and predictors, and report the skipped ones with the reason. ``ModelSpec.from_name``,
  ``drevalpy run`` and ``drevalpy single`` accept a recipe such as
  ``pca[gene_expression]:fingerprints:ridge`` wherever they accepted a zoo name, and the model is
  named by the recipe. See :doc:`/how-to/evaluate_component_grids`.
- **Recorded failures.** A run that raises no longer aborts the experiment. ``drevalpy.run``
  logs a warning, keeps the failure in ``ExperimentResult.failures`` and writes ``failures.json``;
  ``on_error="raise"`` restores the old behavior. ``drevalpy run`` prints the failures and exits
  with status 1, and the report has a "Failed runs" table. If every run fails, ``run`` raises
  ``ExperimentFailedError``. Each record in ``failures.json`` holds the ``identity`` of the failed run
  (model, dataset, split mode, fold, randomization and robustness repeat), the exception type and its message.
- **Stress tests.** A randomization test or a robustness test is a stress test. The number of
  robustness repeats is ``--robustness-repeats`` (``robustness_repeats=``), saved as ``robustness_repeat``;
  1.x called it ``--n_trials_robustness``. Randomizing a graph view keeps the degree of every node.
  See :doc:`/how-to/run_stress_tests`.
- **Curve-quality rule.** ``drevalpy.data.CurveQuality`` describes the rule that ``split`` hands
  to the splitter, with the default unchanged. ``drevalpy run`` and ``drevalpy data split`` take
  ``--no-quality-filter``, ``--min-relevance-score`` and ``--min-abs-fold-change``, and
  ``CurveQuality.off()`` splits a dataset that has only a pEC50 matrix. Each fold records the rule
  under ``quality``. A splitter takes a required ``quality`` keyword and calls
  ``observed_mask(dataset, quality)``; a splitter without it fails at registration. See
  :doc:`/reference/datasets` and :doc:`/how-to/extend/write_splitter`.
- **Weights & Biases.** ``--wandb-project`` on ``drevalpy run`` and ``drevalpy single``, and
  ``wandb_project=`` in Python, log one W&B run per tuning trial and a final run per fold with the
  test metrics.
- **Parallel single-drug models.** ``--n-jobs`` on ``drevalpy run``, ``single`` and ``train``, and
  ``n_jobs=`` in Python, fit the per-drug models of ``SingleDrugElasticNet``,
  ``SingleDrugRandomForest``, ``SingleDrugKNN``, ``MOLIR`` and ``SuperFELTR`` in parallel
  worker processes. The results are identical for every value. ``MOLIR`` and ``SuperFELTR`` train on
  the CPU and write no default Lightning logs.

Components and imports
~~~~~~~~~~~~~~~~~~~~~~

- A featurizer declares the feature format it emits as ``output_format``, and a predictor the format it
  needs on each side as ``sample_input_format`` and ``drug_input_format``. Give them on the class or as
  keywords of ``@register``; the keyword wins. A missing or wrong-typed format raises ``TypeError`` when the
  class is created or registered, and a model whose parts disagree is rejected before training.
  ``FeatureFormat`` is importable from ``drevalpy.plugin``.
- A global feature block is made with ``global_feature_block``, its side information is ``annotations``,
  and a ``BlockSpec`` marks it with ``entity_aligned=False``.
- ``precompute(dataset, featurizer_cls, hyperparameters)`` and ``precompute_all(dataset)`` in
  ``drevalpy.components.featurizers.precompute`` store the featurizer variants, and the sampling of the
  configurations is seeded.
- A file may store a sample view under its public name or under the stored name (``copy_number_variation``
  for ``copy_number_variation_gistic``); the public name wins when both exist.
- ``drevalpy.types`` exports only ``Dataset``, ``SplitMask`` and ``SplitMasks``, and
  ``drevalpy.types.results`` only ``ExperimentResult``, ``ModelResult``, ``RunResult`` and
  ``ExperimentFailedError``. Import every other type from the module that defines it. A ``SplitMask`` is
  read-only and has a public ``pair_seed``.
- A plot declares its ``PlotRequirement`` values, and
  ``drevalpy.visualization.requirements.requirements_met(experiment, requirements)`` checks them against an
  experiment.

Other changes since 1.6.0
~~~~~~~~~~~~~~~~~~~~~~~~~

- A dataset is one ``.h5mu`` file. A model is a featurizer and a predictor, chosen by a zoo
  preset or a recipe such as ``pca[gene_expression]:elasticNet``.
- MultiQC is a regular dependency, so ``drevalpy report`` works without an extra. PyTorch
  comes with the ``cpu``, ``cu126`` or ``cu130`` extra.
- Datasets that you download by name are checked against a pinned ``sha256``. ``TOYv1`` is republished and
  holds every gene of the shipped gene lists, so every preset runs on it.
- Precily downloads the MSigDB C2 CP v6.1 gene sets on first use when a dataset has no
  ``uns["pathways_gmt"]``, and keeps them in the cache directory.
- Datasets that you download by name are cached in the system cache directory, which
  ``DREVALPY_CACHE_DIR`` overrides, instead of a ``--path_data`` directory.
- Curated datasets keep the curve fitted to each replicate dose series in
  ``response.uns["replicate_fits"]``, one row per series, instead of the ``response_rr1`` and
  ``response_rr2`` modalities. Subsetting and ``concat`` carry the table along; see
  :ref:`reference/datasets:File layout`.
- The built-in datasets are the gen5 files, each pinned by its sha256, and ``CCLE``,
  ``PRISM_19Q4`` and ``PRISM_20Q2`` are registered. A replicate is one dose series on one plate,
  and series with fewer than 4 doses are not fitted.
- Extension files, zoo files and plugins add components, split modes, datasets and
  visualizations without editing DrEvalPy. See :doc:`/how-to/extend/index`.

The documentation
~~~~~~~~~~~~~~~~~

The documentation has moved from the CLI and Python guides to four sections: tutorials,
how-to guides, reference and explanation. The tutorials run offline on a synthetic dataset, and
every code block in the tutorials and how-to guides is run by the test suite. The pages use the
pydata-sphinx-theme and have synced command-line and Python tabs. Old page addresses are not
redirected.

📄 Our paper is finally published! :date:`2026-05-12`
--------------------------------------------------------------
Our paper is published in `Nature Communications <https://doi.org/10.1038/s41467-026-72903-w>`__; see :doc:`citing` to cite it.

📄 Our preprint is out! :date:`2025-05-29`
--------------------------------------------------------------
The preprint is on `bioRxiv <https://www.biorxiv.org/content/10.1101/2025.05.26.655288v1>`__.

🚀 We have launched the DrEval Challenge 🚀 :date:`2024-11-26`
--------------------------------------------------------------
Beat the baseline models in DrEval and tell us the result; the best get chocolate.

📜 Origin Story 💊 :date:`2023-11-20`
--------------------------------------------------------------
DrEvalPy was born to test drug response prediction models with fairness and transparency.
