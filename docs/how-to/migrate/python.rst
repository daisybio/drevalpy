Migrate Python scripts from 1.x
===============================

.. interface: python

A 1.x script imported ``MODEL_FACTORY`` and ``drug_response_experiment``, loaded
a ``DrugResponseDataset`` and called the experiment with a list of model
classes. In v2 the models are ``ModelSpec`` objects, the dataset is a
``Dataset`` and ``drevalpy.run`` returns the results instead of only writing
files. This page maps each import, parameter and result file. For command-line
runs see :doc:`cli`; for a model class of your own see :doc:`models`.

A 1.x script, for comparison:

.. snippet: skip

.. code-block:: python

   from drevalpy.datasets.loader import load_dataset
   from drevalpy.experiment import drug_response_experiment
   from drevalpy.models import MODEL_FACTORY

   response_data = load_dataset("GDSC1", path_data="data")
   drug_response_experiment(
       models=[MODEL_FACTORY["ElasticNet"]],
       baselines=[MODEL_FACTORY["NaiveMeanEffectsPredictor"]],
       response_data=response_data,
       test_mode="LCO",
       run_id="my_first_run",
       path_out="results",
       hyperparameter_tuning=False,
   )

The v2 script has no side effects until you call ``save``. It runs offline,
because it uses the synthetic dataset instead of ``GDSC1``:

.. code-block:: python

   import drevalpy

   dataset = drevalpy.synthetic(n_samples=60, n_drugs=8, seed=3)
   models = [
       drevalpy.ModelSpec.from_name(name)
       for name in ["NaiveTissueMeanPredictor", "NaiveMeanEffectsPredictor"]
   ]
   result = drevalpy.run(models, dataset, "LSO", hyperparameter_tuning=False)
   print(result.model_names)

With real data, replace the first line after the import with
``dataset = drevalpy.load("GDSC1")``. It takes a built-in name or the path to a
``.h5mu`` file; see :doc:`/how-to/bring_your_own_data`.

Map imports and objects
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 1.x
     - v2
   * - ``drevalpy.models.MODEL_FACTORY[name]`` (a class)
     - ``drevalpy.ModelSpec.from_name(name)`` (a recipe). Names are in
       :ref:`how-to/migrate/index:Map 1.x model names to v2`. ``MODEL_FACTORY``,
       ``MULTI_DRUG_MODEL_FACTORY`` and ``SINGLE_DRUG_MODEL_FACTORY`` no longer
       exist.
   * - ``drevalpy.experiment.drug_response_experiment``
     - ``drevalpy.run``. It returns an ``ExperimentResult``.
   * - ``drevalpy.datasets.loader.load_dataset`` and
       ``AVAILABLE_DATASETS``
     - ``drevalpy.load(name_or_path)``. The built-in names are ``BeatAML2``,
       ``CTRPv1``, ``CTRPv2``, ``GDSC1``, ``GDSC2``, ``PDX_Bruna`` and
       ``TOYv1``; ``CCLE``, ``TOYv2`` and the ``CTRPv2_clean*`` datasets are gone.
   * - ``DrugResponseDataset`` (``response``, ``cell_line_ids``, ``drug_ids``:
       one row per pair)
     - ``drevalpy.types.data.dataset.Dataset``: ``response_matrix`` of samples by
       drugs, with ``sample_ids`` and ``drug_ids``. See
       :doc:`/reference/datasets`.
   * - ``DrugResponseDataset.from_csv``, ``to_csv``, ``to_dataframe``
     - ``Dataset.load`` and ``Dataset.save`` read and write the ``.h5mu`` file.
       There is no CSV importer; build the dataset as
       :doc:`/how-to/bring_your_own_data` describes.
   * - ``FeatureDataset``, ``load_cell_line_features``,
       ``load_drug_features``
     - Removed. Features are tables in the ``.h5mu`` file, and a featurizer
       component reads them. See :doc:`models`.
   * - ``dataset.split_dataset(split_method=..., n_cv_splits=...)`` and
       ``dataset.cv_splits``
     - ``drevalpy.split(dataset, mode, n_splits=5, validation_ratio=0.1,
       random_state=42)``. It returns a list of ``SplitMasks``, each with the
       boolean masks ``train``, ``val`` and ``test`` over the response matrix.
   * - ``dataset.save_splits(path)``, ``dataset.load_splits(path)``
     - ``drevalpy data split`` writes ``fold_N.npz`` files (:doc:`cli`).
       Old split files do not load.
   * - ``drevalpy.utils.get_response_transformation``
     - Unchanged, in ``drevalpy.utils``. It accepts ``"None"``, ``"standard"``,
       ``"minmax"``, ``"robust"``, ``"drug_mean"`` and ``"drug_tissue_mean"``. The
       last two return a ``drevalpy.utils.response_transform.GroupMeanCenterer``.
   * - ``drevalpy.evaluation.AVAILABLE_METRICS``
     - Unchanged: the same seven metrics.
   * - ``drevalpy.evaluation.evaluate(dataset, metric)``
     - ``evaluate(predictions, response, metric)`` takes two arrays. Passing an
       object that has ``predictions`` and ``response`` attributes still works.
   * - ``drevalpy.visualization.*`` classes such as ``Violin`` and
       ``Heatmap``, and ``drevalpy.pipeline_function``
     - ``drevalpy.visualization.create_report(result, output_dir)``, or the
       ``drevalpy report`` command. Plots are visualization components; see
       :doc:`/how-to/make_report`.
   * - ``--version`` on the command line
     - Unchanged: ``drevalpy --version``. In Python, read ``drevalpy.__version__``.
   * - ``except KeyError`` around ``MODEL_FACTORY[name]``
     - ``ModelSpec.from_name`` raises ``UnknownModelError``, an
       ``UnknownNameError`` from ``drevalpy.errors``. It is also a ``KeyError``
       and a ``ValueError``, so the old ``except`` clause still catches it.
       ``DrEvalError`` is the base class of the library's errors.

Map the ``drug_response_experiment`` parameters
-----------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 1.x parameter
     - v2 ``drevalpy.run`` parameter
   * - ``models``, ``baselines``
     - ``models``: one list of ``ModelSpec`` objects. v2 does not add
       ``NaiveMeanEffectsPredictor`` for you.
   * - ``response_data``
     - ``dataset``
   * - ``test_mode``
     - ``split_mode``, one mode per call. ``"LCO"`` is accepted as an alias of
       ``"LSO"``. The default ``"LPO"`` is gone: the Python function requires
       ``split_mode``.
   * - ``response_transformation``
     - ``response_transformation``. An unfitted sklearn transformer, or
       ``None``, as before.
   * - ``hpam_optimization_metric``
     - ``hpo_metric``
   * - ``hyperparameter_tuning``
     - ``hyperparameter_tuning``. Unchanged name. Tuning is an Optuna search of
       ``hpo_num_trials`` trials (16 by default), not a grid.
   * - ``randomization_mode``
     - ``randomization_modes``
   * - ``randomization_type``
     - ``randomization_type``
   * - ``n_trials_robustness``
     - ``robustness_trials``
   * - ``n_cv_splits``
     - Removed. ``run`` makes five folds. Use ``drevalpy.split`` with
       ``n_splits`` and ``drevalpy.single`` for other counts.
   * - ``run_id``, ``path_out``, ``overwrite``, ``path_data``
     - Removed. ``run`` writes nothing. Save the result yourself
       (see below).
   * - ``multiprocessing``
     - Removed. Run folds in parallel with ``drevalpy.single``
       (:doc:`/how-to/run_folds_in_parallel`).
   * - ``cross_study_datasets``
     - Pass a list of datasets to ``dataset`` and use the split mode ``"LStO"``, or build the
       dataset with ``drevalpy.data.concat``. See :doc:`/how-to/run_cross_study`.
   * - ``n_jobs``
     - ``n_jobs`` of ``drevalpy.run``, ``drevalpy.single`` and ``drevalpy.train``. It fits
       the per-drug models of single-drug models in parallel and does not change the
       results. See :doc:`/how-to/run_experiment`.
   * - ``model_checkpoint_dir``
     - Removed from ``run``. Save a model with ``DRPModel.save``.
   * - ``final_model_on_full_data``
     - ``drevalpy.train(model, dataset)`` trains on all pairs and returns the fitted
       ``DRPModel``. See :doc:`/how-to/train_save_load`.
   * - ``wandb_project``
     - ``wandb_project`` of ``drevalpy.run`` and ``drevalpy.single``. See
       :doc:`/how-to/run_experiment`.
   * - ``custom_splitter``, ``custom_split_name``
     - Removed. Register a splitter component
       (:doc:`/how-to/extend/write_splitter`).

New parameters that 1.x did not have: ``seed`` (default ``42``),
``hpo_num_trials``, ``hpo_random_state`` and ``randomization_random_state``.

Save and read the results
-------------------------

1.x wrote a tree of CSV files under ``path_out/run_id``. In v2 the object that
``run`` returns is the result. Save it to a directory and load it later with
``ExperimentResult.load``:

.. code-block:: python

   from drevalpy.types.results import ExperimentResult

   result.save("my_first_run")
   reloaded = ExperimentResult.load("my_first_run")
   print(reloaded.model_names)

The 1.x result files map to these calls:

- ``evaluation_results.csv``: ``result.summary_table()``, a dictionary of
  ``{model: {metric: mean over folds}}``.
- ``evaluation_results_per_drug.csv``: ``result.summary_table(group_by="drug")``.
  Each metric is computed within each drug of a fold, then averaged.
- ``evaluation_results_per_cl.csv``: ``result.summary_table(group_by="sample")``.
- ``true_vs_pred.csv``: the arrays on each ``RunResult``, shown below.
- ``predictions/predictions_split_N.csv``: one ``RunResult`` per fold in
  ``result.models[i].runs``.
- ``best_hpams/best_hpams_split_N.json``: ``RunResult.best_hyperparameters``.

.. code-block:: python

   import pandas as pd

   print(pd.DataFrame(result.summary_table()).T)
   print(pd.DataFrame(result.summary_table(group_by="drug")).T)

   run = result.models[0].runs[0]
   true_vs_pred = pd.DataFrame(
       {
           "sample": run.sample_ids,
           "drug": run.drug_ids,
           "response": run.ground_truth,
           "prediction": run.predictions,
       }
   )
   print(true_vs_pred.head())
   print(run.best_hyperparameters)

``response`` is a pEC50 value, from the response matrix. 1.x gave ``LN_IC50`` or
the measure you chose. A group where a metric is undefined, such as the Pearson
correlation of a constant prediction, is skipped in the grouped tables and not
scored as 0.

The report that 1.x wrote as ``index.html`` is built from the saved result:

.. code-block:: python

   from drevalpy.visualization import create_report

   create_report(reloaded, "report", reference_model="NaiveMeanEffectsPredictor")

Run one model on one fold
-------------------------

The 1.x scripts for single steps (``train-cv``, ``test-cv``) become
``drevalpy.single``. It takes the masks of one fold from ``drevalpy.split`` and
returns a ``RunResult``:

.. code-block:: python

   folds = drevalpy.split(dataset, "LSO", n_splits=3)
   run = drevalpy.single(models[0], dataset, folds[0], hyperparameter_tuning=False)
   run.save("fold_0.npz")
   print(run.metrics["RMSE"])

Combine the ``RunResult`` files with ``ExperimentResult(run_results)`` or the
``drevalpy aggregate`` command.

Train, save and load a model
----------------------------

In 1.x a model object called ``build_model``, ``train``, ``predict``, ``save``
and ``load``, and ``save`` wrote a model-specific directory. In v2 the model is a
``DRPModel`` made from a ``ModelSpec``. It trains on a mask, predicts on a mask
and saves one ``.zip`` archive:

.. code-block:: python

   import numpy as np
   from drevalpy.models import DRPModel

   fold = folds[0]
   model = DRPModel(models[0])
   model.train(dataset, fold.train, early_stopping_scope=None)
   predictions = model.predict(dataset, fold.test)

   model.save("model.zip")
   restored = DRPModel.load("model.zip")
   print(np.allclose(restored.predict(dataset, fold.test), predictions))

:doc:`/how-to/train_save_load` covers this in full. A checkpoint from 1.x does
not load; see :ref:`how-to/migrate/index:Which 1.x files still load`.

Stress tests
------------

The randomization and robustness tests keep their names. In ``run`` pass
``randomization_modes=["SVCC"]`` and ``robustness_trials=2``. As functions,
``drevalpy.randomization`` and ``drevalpy.robustness`` build the changed
datasets and splits. A robustness test reshuffles the order of the pairs and
keeps the seed. :doc:`/how-to/run_stress_tests` has the details.
