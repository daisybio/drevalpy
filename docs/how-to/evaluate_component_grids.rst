Evaluate component grids
========================

.. interface: both

A model in DrEvalPy is built from a sample featurizer, a drug featurizer and a predictor (see
:doc:`/explanation/models_from_components`). A grid evaluates every valid combination of the
components you list, so you can compare featurizers or predictors in one run instead of writing
one model per pair. This page shows how to build a grid, how to name a single combination as a
recipe, how skipped combinations are reported, and what happens when one run of the grid fails.

The examples use the simulated dataset, so they run offline in a few seconds.

Build a grid
------------

Give the sample featurizers, the drug featurizers and the predictors. DrEvalPy forms every
combination ``<sample>:<drug>:<predictor>``, drops the ones whose components do not fit together,
and runs the rest. A featurizer that needs a view takes it in brackets, as in
``raw[gene_expression]``; see :doc:`/reference/recipes` for the full syntax and
:doc:`/how-to/list_available` for the names.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      ``-S`` (``--sample-featurizer``), ``-D`` (``--drug-featurizer``) and ``-P`` (``--predictor``)
      are repeatable. Predictors are required for a grid; ``-S`` and ``-D`` are optional.

      .. code-block:: bash

         drevalpy data synthetic --output toy.h5mu
         drevalpy run --dataset toy.h5mu --split-mode LSO --no-hpo --output-dir results_grid \
             -S "raw[gene_expression]" -S tissue -D fingerprints -P ridge -P naiveTissueMean

      The command names each skipped combination and the reason, then runs the others and prints
      the summary. The skipped line comes first, before the folds run:

      .. code-block:: text

         Skipped raw[gene_expression]:fingerprints:naiveTissueMean: Predictor 'naiveTissueMean' needs sample block(s) ['tissue'] but the sample featurizer 'raw' emits [gene_expression]
         Wrote experiment results to results_grid
         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LSO
             Normalized by: None
             Models: 3
                 raw[gene_expression]:fingerprints:ridge (5 folds): MSE=0.5534, RMSE=0.7437, MAE=0.5830, R^2=0.7925, Pearson=0.8936, Spearman=0.9025, Kendall=0.7252
                 tissue:fingerprints:ridge (5 folds): MSE=0.7718, RMSE=0.8748, MAE=0.7049, R^2=0.7111, Pearson=0.8499, Spearman=0.8483, Kendall=0.6686
                 tissue:fingerprints:naiveTissueMean (5 folds): MSE=2.6879, RMSE=1.6393, MAE=1.3716, R^2=-0.0076, Pearson=0.1044, Spearman=0.1215, Kendall=0.0932

   .. tab-item:: Python
      :sync: python

      ``drevalpy.models.combinations`` expands the lists. It returns the valid model specs and the
      skipped combinations, so you can look at both before spending time on a run.

      .. code-block:: python

         import drevalpy
         from drevalpy.models import combinations

         grid = combinations(
             sample=["raw[gene_expression]", "tissue"],
             drug=["fingerprints"],
             predictors=["ridge", "naiveTissueMean"],
         )
         print([spec.name for spec in grid.specs])
         for skipped in grid.skipped:
             print(skipped.recipe, "|", skipped.reason)

      .. code-block:: text

         ['raw[gene_expression]:fingerprints:ridge', 'tissue:fingerprints:ridge', 'tissue:fingerprints:naiveTissueMean']
         raw[gene_expression]:fingerprints:naiveTissueMean | Predictor 'naiveTissueMean' needs sample block(s) ['tissue'] but the sample featurizer 'raw' emits [gene_expression]

      Pass ``grid.specs`` to ``run`` like any list of models:

      .. code-block:: python

         dataset = drevalpy.synthetic()
         results = drevalpy.run(grid.specs, dataset, "LSO", hyperparameter_tuning=False)
         print(results)

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LSO
             Normalized by: None
             Models: 3
                 raw[gene_expression]:fingerprints:ridge (5 folds): MSE=0.5534, RMSE=0.7437, MAE=0.5830, R^2=0.7925, Pearson=0.8936, Spearman=0.9025, Kendall=0.7252
                 tissue:fingerprints:ridge (5 folds): MSE=0.7718, RMSE=0.8748, MAE=0.7049, R^2=0.7111, Pearson=0.8499, Spearman=0.8483, Kendall=0.6686
                 tissue:fingerprints:naiveTissueMean (5 folds): MSE=2.6879, RMSE=1.6393, MAE=1.3716, R^2=-0.0076, Pearson=0.1044, Spearman=0.1215, Kendall=0.0932

``naiveTissueMean`` predicts from the tissue block, so it fits ``tissue`` and not ``raw``. The
grid keeps three of the four combinations and reports the fourth instead of failing.
``combinations`` raises ``ValueError`` when nothing is valid, when you give drug featurizers
without sample featurizers, and when the predictor list is empty. The command line turns the same
errors into a usage error.

Every model is named by its full recipe, views included, so the names are unique and the results
directory of the command line has one folder per recipe. If two entries expand to the same recipe,
only the first is kept.

.. note::

   Some predictors need a drug featurizer even when they do not use drug features.
   ``-S tissue -P naiveTissueMean`` without ``-D`` is skipped with "requires a drug_featurizer"
   (and is a usage error if nothing else in the grid is valid). Add ``-D identity``, which
   one-hot encodes the drug IDs, to run it. The same holds on a registered dataset such as GDSC2.

Name one combination as a recipe
--------------------------------

A recipe is the model name of one combination. The ``MODELS`` arguments of ``run`` and
``single``, and ``ModelSpec.from_name``, accept recipes as well as zoo names. A zoo name wins if a
recipe and a zoo entry would have the same name. Use a recipe for a single combination and a grid
for several; both can be mixed on one command line, and a model that appears twice runs once.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy run "raw[gene_expression]:fingerprints:ridge" NaiveDrugMeanPredictor \
             --dataset toy.h5mu --split-mode LSO --no-hpo --output-dir results_recipe

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy import ModelSpec

         spec = ModelSpec.from_name("raw[gene_expression]:fingerprints:ridge")
         print(spec.name)

      .. code-block:: text

         raw[gene_expression]:fingerprints:ridge

A name that is neither a zoo name nor a valid recipe raises ``ValueError`` (``UnknownModelError``)
whose message says that the string is neither and gives the reason the recipe was rejected, for
example the available predictors for an unknown one.

Find out why a run failed
-------------------------

A grid can pass validation and still fail while it runs, for example because the dataset lacks the
view a featurizer asks for. By default one failing run does not stop the others. ``run`` logs a
warning, keeps going and records the failure; the models that worked are scored as usual. The
command line exits with code 1 when any run failed, after saving everything that finished.

The model below asks for a methylation view that the simulated dataset does not have:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command exits with code 1 by design, so the example is not run as a test.

      .. snippet: skip

      .. code-block:: bash

         drevalpy run "raw[methylation]:fingerprints:ridge" NaivePredictor \
             --dataset toy.h5mu --split-mode LSO --no-hpo --output-dir results_failed

      ``NaivePredictor`` is scored and saved. The failures are listed on standard error after the
      summary:

      .. code-block:: text

         5 run(s) failed:
           raw[methylation]:fingerprints:ridge fold 0: KeyError: "Modality 'methylation' not found. Available: ['gene_expression', 'response']"
           raw[methylation]:fingerprints:ridge fold 1: KeyError: "Modality 'methylation' not found. Available: ['gene_expression', 'response']"
           raw[methylation]:fingerprints:ridge fold 2: KeyError: "Modality 'methylation' not found. Available: ['gene_expression', 'response']"
           raw[methylation]:fingerprints:ridge fold 3: KeyError: "Modality 'methylation' not found. Available: ['gene_expression', 'response']"
           raw[methylation]:fingerprints:ridge fold 4: KeyError: "Modality 'methylation' not found. Available: ['gene_expression', 'response']"

      The output directory also holds ``failures.json``; see `The failures file`_. If every run
      fails, nothing is saved but the failures are printed and the exit code is 1. There is no
      command-line switch for ``on_error="raise"``.

   .. tab-item:: Python
      :sync: python

      ``run`` returns the result of the runs that worked, and ``failures`` holds one record per
      failed run.

      .. code-block:: python

         models = [ModelSpec.from_name(name) for name in ["raw[methylation]:fingerprints:ridge", "NaivePredictor"]]
         results = drevalpy.run(models, dataset, "LSO", hyperparameter_tuning=False)
         print(results.model_names)
         print(len(results.failures))
         print(results.failures[0])

      .. code-block:: text

         ['NaivePredictor']
         5
         raw[methylation]:fingerprints:ridge fold 0: KeyError: "Modality 'methylation' not found. Available: ['gene_expression', 'response']"

      A record is a ``RunFailure`` with the ``identity`` of the run that failed (model, dataset,
      split mode, fold, and the randomization and robustness repeat, ``None`` for an ordinary run),
      the exception type and its message. If every run fails there is no result to return, and ``run`` raises
      ``ExperimentFailedError``, which carries the same records in its ``failures``.

      Pass ``on_error="raise"`` to stop at the first error and see the traceback, which is
      useful while debugging a new model:

      .. code-block:: python

         try:
             drevalpy.run(models, dataset, "LSO", hyperparameter_tuning=False, on_error="raise")
         except KeyError as error:
             print("raised:", error)

      .. code-block:: text

         raised: "Modality 'methylation' not found. Available: ['gene_expression', 'response']"

A failed run is not an empty score: it has no entry in the summary at all. A model whose folds
all failed is missing from the table rather than present with zeros, and a model with some failed
folds is scored on the remaining ones, which the fold count in the summary shows. Check
``results.failures`` (or the exit code) before you read a comparison. The HTML report lists the
recorded failures in a "Failed runs" table; see :doc:`/how-to/make_report`.

The failures file
-----------------

``ExperimentResult.save`` writes ``failures.json`` next to ``metadata.json`` when there is at
least one failure, and nothing when there is none. It is a list of the same records, so you can
read it without DrEvalPy:

.. code-block:: text

   [
     {
       "identity": {
         "model_name": "raw[methylation]:fingerprints:ridge",
         "dataset_name": "SYNTHETIC",
         "split_mode": "LSO",
         "fold_index": 0,
         "fold_id": "3f9a1c5e7b20",
         "randomization": null,
         "robustness_repeat": null,
         "provenance": "{}"
       },
       "error_type": "KeyError",
       "message": "\"Modality 'methylation' not found. Available: ['gene_expression', 'response']\""
     },
     ...
   ]

``ExperimentResult.load`` reads the file back into ``failures``. The layout is in
:doc:`/reference/results`.
