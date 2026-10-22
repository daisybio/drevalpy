News
====

.. role:: date
    :class: date

🧬 DrEvalPy 2.0.0 :date:`2026-10-22`
--------------------------------------------------------------
v2 is a rewrite of the interface, not an upgrade in place. It is not released yet:
``pip install drevalpy`` still installs 1.6.0. Nothing that 1.x wrote loads in v2, so re-run
experiments and retrain models. :doc:`/how-to/migrate/index` maps every 1.x option, function and
model to its v2 counterpart. This entry lists what changed for users.

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
- The tuning option ``--hpo-num-samples`` (``hpo_num_samples``) is ``--hpo-trials``
  (``hpo_num_trials``), because "sample" now only means the biological source. Release 1.6.0
  never had the old name.
- A dataset needs a unique observation index and a ``tissue`` column. The columns
  ``cellosaurus_id`` and ``cell_line_name`` are optional and only make sense for datasets whose
  samples are cell lines. See :doc:`/reference/datasets`.
- Model checkpoints are format 4. Loading an older checkpoint raises an error that says to
  retrain the model. Loading a result file that has ``cell_line_ids`` raises an error that says
  to re-run the experiment, and so does a dataset with precomputed featurizer variants stored
  under ``cell_line_featurizer_variants``.

Defaults and results that changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``drevalpy single`` no longer standardizes the response: ``--response-transformation``
  defaults to ``none``. Earlier builds of the v2 branch defaulted to ``standard``; pass
  ``--response-transformation standard`` to keep that behavior. ``drevalpy run`` has the same
  option, with the same default. The values are ``none``, ``standard``, ``minmax`` and
  ``robust``.
- ``ExperimentResult.summary_table`` is a method, not a property. Call ``summary_table()`` for
  the pooled metrics, or ``summary_table(group_by="drug")`` or ``summary_table(group_by="sample")``
  to compute each metric within every drug or sample and average over the groups. A group where
  the metric is undefined, such as the Pearson correlation of a constant prediction, is skipped
  and never counted as 0. ``summary_group_counts`` returns how many groups each value is based
  on.
- ``drevalpy run`` and ``drevalpy aggregate`` take ``--group-by drug|sample`` and print that
  table. The report has a second summary table, "Model Summary Table per Drug".
- One ``--seed`` (``seed=`` in Python, default 42) drives the splits, the hyperparameter search,
  the randomization tests and the seeding of every model fit.
- After tuning, the final model now trains with the tuned value of a power-of-two
  hyperparameter (``fingerprints.n_bits``, ``bpe_pharmaformer.max_length``). Earlier builds of
  the v2 branch trained it with the exponent, for example 9 fingerprint bits instead of 512, and
  stored that value in ``best_hyperparameters`` and the checkpoint. Re-run tuned experiments
  that use these featurizers.
- A dataset that ``drevalpy experiments randomization`` writes now trains through
  ``drevalpy single``. The file keeps its randomization label, so the run records it and
  ``drevalpy aggregate`` keeps it apart from the ordinary runs. Earlier builds of the v2 branch
  could not train these files with ``single``.
- A randomization test that would change nothing is skipped with a warning and produces no
  run: ``SVRD`` or ``SVCD`` on a dataset without the drug view, and ``SVCC`` or ``SVCD`` on a
  model with a single view. Earlier builds of the v2 branch trained on an unchanged copy of the
  dataset and labelled the run as randomized. ``experiments randomization`` then writes 0 files
  and exits with status 0, and ``Dataset.with_randomized_views`` raises ``KeyError`` for a view
  the dataset does not store.

New commands and functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``drevalpy.synthetic()`` and ``drevalpy data synthetic --output toy.h5mu`` build a small
  simulated dataset, by default 60 samples, 8 drugs and 6 tissues with a gene expression view.
  It needs no download, and its drugs differ so much in average potency that pooled scores
  look good for models that rank samples poorly. ``drevalpy.testing.build_synthetic_dataset``
  is an alias.
- ``drevalpy data split`` and ``drevalpy experiments randomization`` accept a registered
  dataset name as well as a file, like ``drevalpy run --dataset``. ``drevalpy single`` reads
  only a file.
- ``drevalpy report --dataset`` is accepted for pipeline compatibility; the report does not
  read the file.
- ``drevalpy list zoo-presets`` and ``drevalpy list datasets`` show the model presets you can
  pass to ``run`` and the datasets you can load by name.
- ``DREVALPY_EXTENSIONS_DIR`` is read by ``load_extensions()`` in Python, not only by the
  command line. It can name several directories, separated by ``:`` on Linux and macOS.
- ``register_source`` and ``register_dataset`` raise ``ValueError`` when the name is already
  registered with a different entry, including a built-in one. Pass ``override=True`` to
  replace it. Registering an identical entry again does nothing.

Other changes since 1.6.0
~~~~~~~~~~~~~~~~~~~~~~~~~

- A dataset is one ``.h5mu`` file. A model is a featurizer and a predictor, chosen by a zoo
  preset or a recipe such as ``pca[gene_expression]:elasticNet``. ``MODEL_FACTORY`` and
  ``construct_model`` are gone; use ``ModelSpec.from_name`` and ``drevalpy.run``.
- ``drevalpy`` is the only script. Its subcommands are ``run``, ``single``, ``aggregate``,
  ``curate``, ``report``, ``data``, ``experiments`` and ``list``; the 17 ``drevalpy-*`` scripts
  are removed.
- Hyperparameter tuning is a serial Optuna search. Ray and the ``multiprocessing`` extra are
  gone; run folds in parallel with ``drevalpy single`` instead.
- MultiQC is a regular dependency, so ``drevalpy report`` works without an extra. PyTorch
  comes with the ``cpu``, ``cu126`` or ``cu130`` extra.
- Datasets that you download by name are cached in the system cache directory, which
  ``DREVALPY_CACHE_DIR`` overrides, instead of a ``--path_data`` directory.
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
After a long, long journey, our paper is finally published at `Nature Communications <https://doi.org/10.1038/s41467-026-72903-w>`__!
You can now officially cite us with:

.. note::

    **Critical evaluation of drug response prediction models with DrEval.**
    Judith Bernett,  Pascal Iversen,  Mario Picciani,  Mathias Wilhelm,  Katharina Baum,  Markus List.
    *Nat Commun* **17**, 4238 (2026); doi: https://doi.org/10.1038/s41467-026-72903-w

📄 Our preprint is out! :date:`2025-05-29`
--------------------------------------------------------------
Check out our preprint on `bioRxiv <https://www.biorxiv.org/content/10.1101/2025.05.26.655288v1>`_!

.. note::

    **From Hype to Health Check: Critical Evaluation of Drug Response Prediction Models with DrEval.**
    Judith Bernett,  Pascal Iversen,  Mario Picciani,  Mathias Wilhelm,  Katharina Baum,  Markus List.
    *bioRxiv* 2025.05.26.655288; doi: https://doi.org/10.1101/2025.05.26.655288


🚀 We have launched the DrEval Challenge 🚀 :date:`2024-11-26`
--------------------------------------------------------------

| We think that it is possible to design a meaningful drug response prediction model that outperforms simple baseline models.
|
| The DrEval challenge is simple:
|
| 1. Integrate your model into DrEval by following the Contributor Guide.
| 2. Compare your model to the baseline models provided in the DrEval package, either using the standalone or the Nextflow pipeline.
| 3. Let us know your results!
|    3.1. 🎊If you significantly outperform the RandomForest baseline model in the LSO setting or the GradientBoosting model in the LDO setting, we will personally send you chocolate or another snack of your choosing 🍫.
|    3.2. 🥺If you perform significantly worse than the NaiveDrugMeanPredictor, you will have to send us chocolate.

📜 Origin Story 💊 :date:`2023-11-20`
--------------------------------------------------------------

    *Long ago, the people of science lived in harmony. Researchers collaborated, data flowed freely, and models were tested with integrity.
    But then, the H-Index Nation attacked.
    Suddenly, impact factors ruled all, flashy results overshadowed rigorous testing, and biased benchmarks spread like wildfire. Science, once a beacon of knowledge, became clouded by competition and questionable practices.
    Only a fair and unbiased framework could restore balance.
    And when the field needed it most—drevalpy was born.
    A framework designed to test drug response prediction models with fairness and transparency, cutting through bias and restoring the integrity of scientific evaluation.
    Though the fight against bad practices is long, with drevalpy, balance may yet be restored.*

..
   <!-- DrEvalPy is on PyPI :date:`2024-05-29` -->
   --------------------------------------
