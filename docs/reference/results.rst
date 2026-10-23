Results layout
===============

``drevalpy run`` and ``drevalpy aggregate`` write an ``ExperimentResult`` directory.
``drevalpy single`` and ``drevalpy.single`` produce one ``RunResult`` file.
``ExperimentResult.load(directory)`` reads a directory and ``ExperimentResult.save(directory)``
writes one. ``RunResult.load(path)`` and ``RunResult.save(path)`` do the same for a file.

Experiment directory
--------------------

.. code-block:: text

   <output dir>/
     metadata.json
     failures.json       (only when a run failed)
     <ModelName>/
       metadata.json
       fold_0.npz
       fold_1.npz
       ...

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - Content
   * - ``<output dir>/metadata.json``
     - ``dataset_name``, ``split_mode``, ``normalized_by`` (``null`` unless normalized) and
       ``models``, the list of model names
   * - ``<output dir>/failures.json``
     - Written only when at least one run failed; see `Recorded failures`_
   * - ``<ModelName>/metadata.json``
     - ``model_name``, ``dataset_name``, ``n_folds``, ``n_runs`` and ``aggregate_metrics``
   * - ``<ModelName>/fold_<i>.npz``
     - One run; see below

``aggregate_metrics`` maps each metric to its ``mean`` and ``std`` across the ordinary runs of
the model. Ordinary runs are those without randomization and without a robustness trial.

An experiment holds one dataset name and one split mode; ``ExperimentResult`` rejects runs
that disagree. A model that comes from a grid or a recipe is named by its recipe, such as
``raw[gene_expression]:fingerprints:ridge``, and so is its directory. The dataset name of a
concatenated dataset is the study names joined by ``+``.

Recorded failures
~~~~~~~~~~~~~~~~~

``drevalpy.run`` keeps going when a run raises and keeps a ``RunFailure`` for it, in the tuple
``ExperimentResult.failures``. ``ExperimentResult.save`` writes the records to ``failures.json``
only when there is at least one, and ``ExperimentResult.load`` reads them back. The file is a
JSON list; each entry has these keys:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Content
   * - ``model_name``
     - The model whose run failed
   * - ``fold_index``
     - The fold of the run
   * - ``randomization``
     - ``[mode, view]`` of the randomized dataset, or ``null``
   * - ``robustness_trial``
     - The index of the robustness variant, or ``null`` for an ordinary run
   * - ``error_type``
     - The class name of the exception, such as ``KeyError``
   * - ``message``
     - ``str(exception)``

A failed run writes no ``fold_<i>.npz`` file. The runs of the other models and folds are saved
as usual. ``drevalpy run`` exits with status 1 when ``failures`` is not empty, and
``drevalpy.run(..., on_error="raise")`` raises the first error instead of recording it; see
:doc:`/how-to/evaluate_component_grids`.

Which run is ``fold_<i>``
~~~~~~~~~~~~~~~~~~~~~~~~~

The number ``<i>`` counts the runs of a model in the order they were written. It is not the
fold index; the fold index of a run is in its ``_metadata``. ``drevalpy run`` writes, for each
fold in turn, the original split first and then each robustness variant of it. For every one
of these splits it writes the run on the original dataset, followed by one run per randomized
dataset.

For example, take two models, five folds, ``--robustness-trials 1`` and ``-r SVRC -r SVCC``.
A model with one sample view gets 5 folds x 2 splits x 2 datasets = 20 files: the original
dataset and the ``SVRC`` dataset. ``SVCC`` would randomize nothing when there is a single view,
so it is skipped with a warning and gives no dataset. A model with two sample views gets both
``SVRC`` datasets and both ``SVCC`` datasets, 5 x 2 x 5 = 50 files. A model without views gets
no randomized datasets and 5 x 2 = 10 files.

``drevalpy aggregate RESULTS... -o DIR`` loads run ``.npz`` files and writes the same layout.
Its output directory defaults to ``experiment_results`` and the file numbers follow the
order of the inputs.

Run file
--------

A run file is a compressed ``.npz`` archive.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Array
     - Content
   * - ``predictions``
     - Predicted response of each test pair
   * - ``ground_truth``
     - Observed response, in the same order
   * - ``sample_ids``, ``drug_ids``
     - The sample and drug of each pair, as strings
   * - ``trial_<i>_predictions``
     - Predictions of tuning trial ``i``; present only when the run has tuning trials
   * - ``_metadata``
     - A JSON string with the keys below

``RunResult.load`` raises ``ValueError`` for a file that has ``cell_line_ids`` and no
``sample_ids``, which an earlier version of DrEvalPy wrote.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``_metadata`` key
     - Content
   * - ``model_name``, ``dataset_name``, ``split_mode``
     - Names
   * - ``fold_index``, ``fold_id``
     - The fold of the split; see :doc:`split_modes`
   * - ``best_hyperparameters``
     - The values used for the final fit, under short names; see :doc:`hyperparameters`
   * - ``metrics``
     - The seven metrics of this run; see :doc:`metrics`
   * - ``fold_metadata``
     - The split fields, described below
   * - ``randomization``
     - ``[mode, view]``, or ``null`` for an unrandomized dataset
   * - ``trials``
     - For each tuning trial its ``hyperparameters``, ``metrics`` and ``optimization_metric``,
       or ``null``

``fold_metadata`` holds the metadata of the fold (``mode``, ``fold_index``, ``n_splits``,
``validation_ratio``, ``random_state``, ``dataset``, ``split_mode``, ``fold_id``) and, where
they apply:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Content
   * - ``quality``
     - The curve-quality rule the split applied; see :doc:`split_modes`
   * - ``test_study``
     - The study of the test set; ``LStO`` only
   * - ``n_splits``
     - For ``LStO``, the number of studies
   * - ``robustness_trial``
     - The index of the robustness variant
   * - ``randomization_provenance``
     - ``root_seed``, ``effective_seed``, ``type`` and ``views`` of the randomization
   * - ``execution``
     - How the run was made: ``config`` (the model configuration), ``scopes``, ``hpo`` and
       ``seed``

In ``execution``, ``scopes`` maps each of ``training``, ``stopping``, ``validation``,
``final_training`` and ``test`` to ``null`` or to an object with ``shape``, ``count``,
``pair_seed`` and ``sha256``. ``hpo`` holds ``enabled``, ``metric``, ``num_trials`` and
``random_state``.

Split file
----------

``drevalpy data split`` and ``drevalpy experiments robustness`` write fold files with the
arrays ``train``, ``test``, ``val``, ``_pair_seeds`` and ``_metadata``. They are the input of
``drevalpy single``. :doc:`split_modes` describes the contents.

Report
------

``drevalpy report EXPERIMENT_DIR -o DIR`` reads an experiment directory and writes a MultiQC
report into ``DIR``, which defaults to ``report``. It never reads the dataset, so it has no
``--dataset`` option.
