Choose models for an experiment
===============================

.. interface: both

This page shows how to pick the models for an experiment: which baselines to include, how to
find a model that fits your data, how to select it by name, and how to compose a model that is
not in the zoo.

The examples use the simulated dataset, so they run offline. :doc:`/how-to/list_available`
shows how to print every name used here.

Start with baselines
--------------------

A score means little without a reference. Include at least one **naive baseline**: a predictor
that uses no features beyond the identities of the sample, the drug or the tissue. A model has
to beat the baseline that knows the same entity it is evaluated on.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Zoo name
     - Prediction
   * - ``NaivePredictor``
     - The mean response of the training pairs
   * - ``NaiveDrugMeanPredictor``
     - The mean response of the pair's drug
   * - ``NaiveSampleMeanPredictor``
     - The mean response of the pair's sample
   * - ``NaiveTissueMeanPredictor``
     - The mean response of the pair's tissue
   * - ``NaiveTissueDrugMeanPredictor``
     - The mean response of the pair's tissue-drug combination
   * - ``NaiveMeanEffectsPredictor``
     - The training mean plus effects for the pair's sample and drug

A baseline that looks up an entity falls back to the global mean when the entity is not in the
training set. The split mode decides when that happens: in leave-drug-out (``LDO``), no test
drug has been seen, so ``NaiveDrugMeanPredictor`` predicts the same as ``NaivePredictor``. The
same holds for ``NaiveSampleMeanPredictor`` in ``LSO`` and ``NaiveTissueMeanPredictor`` in
``LTO``. Run the baselines that suit your split mode, as in this example:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command line selects models by zoo name only.

      .. code-block:: bash

         drevalpy data synthetic --output toy.h5mu
         drevalpy run NaivePredictor NaiveDrugMeanPredictor \
             --dataset toy.h5mu --split-mode LDO --no-hpo

      The summary at the end shows identical scores:

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LDO
             Normalized by: None
             Models: 2
                 NaivePredictor (5 folds): MSE=3.0137, RMSE=1.6920, MAE=1.4340, R^2=-1.0724, Pearson=0.0000, Spearman=0.0000, Kendall=0.0000
                 NaiveDrugMeanPredictor (5 folds): MSE=3.0137, RMSE=1.6920, MAE=1.4340, R^2=-1.0724, Pearson=0.0000, Spearman=0.0000, Kendall=0.0000

   .. tab-item:: Python
      :sync: python

      ``ModelSpec.from_name`` turns a zoo name into a model.

      .. code-block:: python

         import drevalpy
         from drevalpy import ModelSpec

         dataset = drevalpy.synthetic()
         baselines = [ModelSpec.from_name(name) for name in ["NaivePredictor", "NaiveDrugMeanPredictor"]]
         results = drevalpy.run(baselines, dataset, "LDO", hyperparameter_tuning=False)
         print(results)

      The ``print`` shows the same scores for both models:

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LDO
             Normalized by: None
             Models: 2
                 NaivePredictor (5 folds): MSE=3.0137, RMSE=1.6920, MAE=1.4340, R^2=-1.0724, Pearson=0.0000, Spearman=0.0000, Kendall=0.0000
                 NaiveDrugMeanPredictor (5 folds): MSE=3.0137, RMSE=1.6920, MAE=1.4340, R^2=-1.0724, Pearson=0.0000, Spearman=0.0000, Kendall=0.0000

Find a model that fits your data
--------------------------------

Every model other than a baseline is a **zoo preset**: a recipe of a sample featurizer, a drug
featurizer and a predictor. Ask three questions of a preset.

**Does the dataset hold the features it reads?**
   The sample featurizer reads one or more views of the dataset, such as ``gene_expression`` or
   ``methylation``. ``scaledGeneExpression``, for example, reads ``gene_expression``. A preset
   whose view the dataset lacks fails when it trains, with an error that lists the views the
   dataset has. Check the dataset before you start: the ``Modalities`` lines of a printed
   dataset show its views, as in :doc:`/how-to/load_and_split`. The drug featurizer
   ``fingerprints`` needs the SMILES of the drugs, which datasets carry.

**Is it a single-drug or a multi-drug model?**
   A multi-drug model trains once on all drugs. A single-drug model, such as
   ``SingleDrugElasticNet``, fits one estimator per drug and uses only the features of the
   samples. ``Scope`` in ``drevalpy list zoo-presets`` shows which a preset is.

**What does it need installed or downloaded?**
   Predictors built on PyTorch, such as the neural networks, need one of the PyTorch extras
   from :doc:`/how-to/install`. Some featurizers, such as ``chemberta`` and ``molgnet``, fetch
   pre-trained artifacts on first use; see :doc:`/reference/environment`.

:doc:`/reference/model_zoo` describes every preset, and :doc:`/reference/components` the
featurizers and predictors behind them. Look up one preset to see its parts:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy list zoo-presets RandomForest

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy.models import zoo

         config = zoo.get_zoo_config("RandomForest")
         print(config.scope.value, config.sample_featurizer.name, config.predictor.name)

      .. code-block:: text

         multi_drug scaledGeneExpression randomForest

Models with hyperparameters are tuned unless you turn tuning off, which adds one fit per trial. :doc:`/how-to/tune_hyperparameters` shows how to keep that cheap.

Compose a model that is not in the zoo
--------------------------------------

Combine the parts yourself when no preset fits, for example a different featurizer on a
different view. A recipe names the parts: ``pca[gene_expression]:fingerprints:ridge`` reads as a
PCA of the ``gene_expression`` view, Morgan fingerprints of the drugs, and a ridge regression.
:doc:`/reference/recipes` has the full syntax.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command line accepts zoo names only, so give your combination a name. Put a YAML file
      in an extension directory, here ``my_models/PcaRidge.yaml``:

      .. code-block:: yaml

         sample_featurizer: pca[gene_expression]
         drug_featurizer: fingerprints
         predictor: ridge

      The file name is the zoo name. Load the directory with the global ``-e`` option, then use
      the name like any other:

      .. snippet: skip

      .. code-block:: bash

         drevalpy -e my_models run NaivePredictor PcaRidge \
             --dataset toy.h5mu --split-mode LDO --no-hpo

      See :doc:`/how-to/extend/load_extensions` for extension directories and
      :doc:`/reference/extension_files` for the file format. A zoo name may not reuse a built-in
      zoo name.

   .. tab-item:: Python
      :sync: python

      Wrap the recipe in a ``ModelSpec`` with a name of your choice:

      .. code-block:: python

         from drevalpy.models.config import from_spec

         ridge = ModelSpec("PcaRidge", from_spec("pca[gene_expression]:fingerprints:ridge"))
         results = drevalpy.run([baselines[0], ridge], dataset, "LDO", hyperparameter_tuning=False)
         print(results)

      The ``print`` shows that the model beats the baseline, which is the point of including
      the baseline:

      .. code-block:: text

         ExperimentResult
             Dataset: SYNTHETIC
             Split mode: LDO
             Normalized by: None
             Models: 2
                 NaivePredictor (5 folds): MSE=3.0137, RMSE=1.6920, MAE=1.4340, R^2=-1.0724, Pearson=0.0000, Spearman=0.0000, Kendall=0.0000
                 PcaRidge (5 folds): MSE=2.6447, RMSE=1.5637, MAE=1.3321, R^2=-0.7880, Pearson=0.4109, Spearman=0.3940, Kendall=0.2949

To write a featurizer or a predictor of your own, see :doc:`/how-to/extend/write_predictor` and
:doc:`/tutorials/own_predictor`.
