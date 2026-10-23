Run a cross-study evaluation
============================

.. interface: both

A cross-study evaluation trains a model on the pairs of some studies and tests it on a study it has
never seen. DrEvalPy does this in two steps: ``concat`` joins the datasets of several studies into
one dataset, and the ``LStO`` split mode (leave one study out) holds out one study per fold. This
page shows both steps, how to score each study separately, and what to watch for when the studies
measured the response on different scales. For what ``LStO`` guarantees, see
:doc:`/reference/split_modes`; for why a cross-study score differs from a within-study one, see
:doc:`/explanation/evaluation_design`.

The runnable examples build two small studies with ``drevalpy.synthetic(name=...)``, so they work
offline. The registered datasets (GDSC2 and CTRPv2) are marked as network snippets; they download
data and need several gigabytes of memory (see `Plan the memory`_).

Combine the datasets of several studies
---------------------------------------

``concat`` takes two or more datasets, one per study, and returns one dataset. Each input is a
study, named by its dataset name, so the names must be distinct and must not contain ``::``. Every
sample becomes one row per study with the ID ``<study>::<sample_id>``, so a sample measured in
both studies is kept twice, once under each prefix. ``obs`` gains the columns ``study`` and
``source_sample_id``. The drug columns are the union of the drug IDs, and a pair that a study did
not measure stays empty. The result is an ordinary dataset: every command and function that takes
a dataset takes it.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      ``drevalpy data concat`` takes the datasets (registered names or ``.h5mu`` paths) and the
      output path. The command prints the name of the result, which is the input names joined
      by ``+``; ``--name`` overrides it.

      The offline examples write two small synthetic studies first. Each needs its own
      ``--name``, because the name becomes the study:

      .. code-block:: bash

         drevalpy data synthetic --name SCREEN_A --output screen_a.h5mu
         drevalpy data synthetic --name SCREEN_B --seed 1 --output screen_b.h5mu
         drevalpy data concat screen_a.h5mu screen_b.h5mu studies.h5mu

      The registered studies are given by name:

      .. snippet: network

      .. code-block:: bash

         drevalpy data concat GDSC2 CTRPv2 gdsc2_ctrpv2.h5mu

   .. tab-item:: Python
      :sync: python

      ``drevalpy.data.concat`` takes a list of datasets. The second study here leaves 30% of its
      pairs unmeasured, so the two studies cover different pairs of the same underlying screen,
      the way two agreeing screens would.

      .. code-block:: python

         import drevalpy
         from drevalpy.data import concat

         screen_a = drevalpy.synthetic(name="SCREEN_A")
         screen_b = drevalpy.synthetic(name="SCREEN_B", missing_fraction=0.3)
         combined = concat([screen_a, screen_b])
         print(combined)
         print(combined.mdata.obs[["study", "source_sample_id"]].head(3))

      .. code-block:: text

         Dataset
             Name: SCREEN_A+SCREEN_B
             Samples: 120
             Drugs: 8
             Measured pairs: 792
             Randomization: None
             Modalities:
                 gene_expression: 120 × 16
                                 study source_sample_id
         SCREEN_A::SYNTH-000  SCREEN_A        SYNTH-000
         SCREEN_A::SYNTH-001  SCREEN_A        SYNTH-001
         SCREEN_A::SYNTH-002  SCREEN_A        SYNTH-002

      The inputs are not changed. ``combined`` is a dataset like any other, so it can be saved
      with ``combined.save("studies.h5mu")`` and loaded again.

      The registered studies are loaded by name first:

      .. snippet: network

      .. code-block:: python

         gdsc2_ctrpv2 = concat([drevalpy.load("GDSC2"), drevalpy.load("CTRPv2")])

``concat`` is strict about what the studies share, because a mismatch would silently mix columns
that mean different things. It raises ``ValueError`` if

- fewer than two datasets are given, or two have the same name;
- an input is a randomized dataset, or already has a ``study`` or ``source_sample_id`` column;
- one drug ID has two different SMILES strings in the inputs (the message says to use curated
  files that share the ID);
- a view has different feature names in two studies (the default ``join="exact"``), or the
  units of a view differ.

Precomputed featurizer variants and the embeddings in ``obsm`` and ``varm`` are not carried over;
``concat`` warns and the featurizers compute them again from the combined views.

Keep only the shared features
-----------------------------

When the studies measured different feature sets, for example different gene panels, ask for the
shared features with ``join="inner"`` (``--join inner`` on the command line). A view that only
some of the studies have is dropped with a warning. A unit mismatch still raises.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. snippet: network

      .. code-block:: bash

         drevalpy data concat GDSC2 CTRPv2 gdsc2_ctrpv2_shared.h5mu --join inner

   .. tab-item:: Python
      :sync: python

      The third study here measured only the first 12 genes of the first study's view. The default ``join="exact"``
      refuses it, naming the view, the two studies and some of the missing features:

      .. code-block:: python

         genes = screen_a.get_sample_feature_names("gene_expression")
         narrow = drevalpy.synthetic(name="SCREEN_C", feature_names=genes[:12])
         try:
             concat([screen_a, narrow])
         except ValueError as error:
             print(str(error).split(" (e.g.")[0])
         shared = concat([screen_a, narrow], join="inner")
         print(shared.name, shared.mdata["gene_expression"].shape)

      .. code-block:: text

         View 'gene_expression' has different feature names in SCREEN_A and SCREEN_C: 11640 features of SCREEN_A missing in SCREEN_C
         SCREEN_A+SCREEN_C (120, 12)

Leave one study out
-------------------

``LStO`` makes one fold per study. The fold tests on every pair of that study and trains on the
pairs of the other studies, so the number of folds is the number of studies. The validation set, used for tuning and
early stopping, is a random share of the training pairs. Pass the combined dataset to ``run``, or
pass several datasets and let ``run`` concatenate them:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      Repeat ``--dataset``. The datasets (registered names or ``.h5mu`` paths, such as
      ``--dataset GDSC2 --dataset CTRPv2``) are concatenated with ``join="exact"`` before the split.

      .. code-block:: bash

         drevalpy run NaiveDrugMeanPredictor NaiveMeanEffectsPredictor \
             --dataset screen_a.h5mu --dataset screen_b.h5mu --split-mode LStO --no-hpo --output-dir results_lsto

      To run on a file written by ``data concat`` (for example one with ``--join inner``), give the
      file as the one ``--dataset``:

      .. code-block:: bash

         drevalpy run NaiveDrugMeanPredictor NaiveMeanEffectsPredictor \
             --dataset studies.h5mu --split-mode LStO --no-hpo --output-dir results_lsto_file

   .. tab-item:: Python
      :sync: python

      ``run`` takes the dataset, a registered name, or a list of them.

      .. code-block:: python

         from drevalpy import ModelSpec

         models = [ModelSpec.from_name(name) for name in ["NaiveDrugMeanPredictor", "NaiveMeanEffectsPredictor"]]
         results = drevalpy.run(models, combined, "LStO", hyperparameter_tuning=False)
         print(results)

      .. code-block:: text

         ExperimentResult
             Dataset: SCREEN_A+SCREEN_B
             Split mode: LStO
             Normalized by: None
             Models: 2
                 NaiveDrugMeanPredictor (2 folds): MSE=0.7859, RMSE=0.8865, MAE=0.7182, R^2=0.7089, Pearson=0.8420, Spearman=0.8450, Kendall=0.6833
                 NaiveMeanEffectsPredictor (2 folds): MSE=0.6905, RMSE=0.8310, MAE=0.6679, R^2=0.7442, Pearson=0.8627, Spearman=0.8670, Kendall=0.6834

      ``drevalpy.run(models, [screen_a, screen_b], "LStO")`` gives the same result. A list of one
      dataset is used as it is, so a dataset that is already a concatenation can be passed in a list.

The folds are ordered by study name, so with GDSC2 and CTRPv2 fold 0 tests on CTRPv2. With three
studies, each fold trains on two and tests on one. ``LStO`` raises ``KeyError`` on a dataset
without a ``study`` column (the message points to ``concat``) and ``ValueError`` with fewer than
two studies. Any other split mode also works on a concatenated dataset; it then mixes the studies
in every fold. The built-in split modes, ``LStO`` included, apply the curve-quality rule; see
:doc:`/reference/datasets`.

Read the score of each study
----------------------------

Under ``LStO`` every fold has exactly one test study, so the fold is the study. The name of the
test study is in the metadata of the fold, and the metrics of the fold are the scores on that
study:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command line prints one row per model, averaged over the folds. To see each study,
      load the saved results in Python with ``drevalpy.types.results.ExperimentResult.load("results_lsto")`` and read
      the folds as in the Python tab.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         for model in results.models:
             for run in model.runs:
                 print(model.model_name, run.fold_metadata["test_study"], round(run.metrics["Pearson"], 3))

      .. code-block:: text

         NaiveDrugMeanPredictor SCREEN_A 0.845
         NaiveDrugMeanPredictor SCREEN_B 0.839
         NaiveMeanEffectsPredictor SCREEN_A 0.863
         NaiveMeanEffectsPredictor SCREEN_B 0.862

``--group-by study`` on the command line and ``summary_table(group_by="study")`` in Python compute
each metric within every study of a fold, then average. The study of a pair is the ``<study>::``
prefix of its sample ID. Under ``LStO`` a fold has one test study, so this equals the pooled
summary. The grouping pays off when a split mode mixes studies in the test set, such as ``LPO`` on
the concatenated dataset:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy run NaiveDrugMeanPredictor NaiveMeanEffectsPredictor \
             --dataset studies.h5mu --split-mode LPO --no-hpo --group-by study \
             --output-dir results_lpo_studies

      The extra table has one row per model and, in brackets, the number of studies scored.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         lpo = drevalpy.run(models, combined, "LPO", hyperparameter_tuning=False)
         by_study = lpo.summary_table(group_by="study")
         print({name: round(row["Pearson"], 4) for name, row in by_study.items()})
         print(lpo.summary_group_counts("study")["NaiveDrugMeanPredictor"]["Pearson"])

      .. code-block:: text

         {'NaiveDrugMeanPredictor': 0.8406, 'NaiveMeanEffectsPredictor': 0.8205}
         2.0

      ``summary_group_counts`` reports how many studies each score is based on. A dataset whose
      sample IDs have no ``<study>::`` prefix has no studies, and the score is ``NaN``.

The HTML report adds a per-study section to the comparison scatter plot for concatenated datasets;
see :doc:`/how-to/make_report`.

Plan the memory
---------------

Both steps hold every study in memory at once. For GDSC2 and CTRPv2, ``concat`` peaks at about
2.4 GB and a ``run`` over the concatenated dataset at about 5.4 GB, before any model-specific
cost. Concatenate once, save the file, and give it to ``run`` as the dataset when you run several
experiments, rather than concatenating in every process.

Mind the response scale
-----------------------

Studies do not measure the response on the same scale. The registered GDSC2 pEC50 values average
about 6.31 and the CTRPv2 values about 5.55. A model trained on one study and tested on the other
predicts on the training scale, so its errors (MSE, RMSE, MAE) and R² contain that offset, and a
low value does not mean the model ranks pairs badly. Compare models on the correlations (Pearson,
Spearman, Kendall), which do not change when a constant is added to the predictions, and read
the error metrics only relative to a baseline run on the same folds, such as
``NaiveMeanEffectsPredictor``.

``--response-transformation`` (``response_transformation`` in Python) scales the response for
scale-sensitive models. It is fitted on the training pairs only, so it does not remove an offset
between the training study and the test study. It does not change what a cross-study score means.
See :doc:`/how-to/run_experiment`.
