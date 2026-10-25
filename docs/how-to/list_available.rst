List what is available
======================

.. interface: both

DrEvalPy registers its models, components, split modes, plots and datasets by name. This page
shows how to list each kind, look at one entry, and see which plugins loaded. The same lists
tell you which names the other commands accept.

``drevalpy list`` has eight subcommands:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Subcommand
     - Lists
   * - ``zoo-presets``
     - Ready-made models, selected by their zoo name
   * - ``predictors``
     - Predictors, the last part of a recipe
   * - ``sample-featurizers``
     - Featurizers for samples, the first part of a recipe
   * - ``drug-featurizers``
     - Featurizers for drugs, the middle part of a recipe
   * - ``splitters``
     - Split modes
   * - ``visualizations``
     - Plots for the report
   * - ``datasets``
     - Built-in and registered datasets
   * - ``plugins``
     - Installed plugin packages

Each list includes whatever your extension directories and installed plugins add; see
:doc:`/how-to/extend/load_extensions`. All commands below print rich tables that adapt to the
width of your terminal; the tables on this page are shortened.

List models
-----------

Models have two layers. A **zoo preset** is a ready-made model with a **zoo name**; you pass
that name to ``run``. Behind it is a recipe of three components: a sample featurizer, a drug
featurizer and a predictor. :doc:`/how-to/choose_models` explains how to pick a model, and
:doc:`/reference/model_zoo` and :doc:`/reference/recipes` describe the presets and the recipe
syntax.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy list zoo-presets

      The table has 33 rows, one per zoo name, with the recipe, scope, prediction mode and
      origin of each:

      .. code-block:: text

         Zoo presets
          Name                   Recipe                                      Scope        Prediction mode   Origin
          ──────────────────────────────────────────────────────────────────────────────────────────────────────────
          AdaBoostDecisionTree   scaledGeneExpression:fingerprints:adaboost  multi_drug   regression        built-in
          DIPK                   concatFeaturizers:molgnet:dipk              multi_drug   regression        built-in
          DrugGNN                landmarkGenes:drugGraph:drugGNN             multi_drug   regression        built-in
          ElasticNet             scaledGeneExpression:fingerprints:elasticNet multi_drug  regression        built-in
          EnsembleMF             scaledGeneExpression:fingerprints:ensembleMF multi_drug  regression        built-in
          ...

      The three parts of a recipe come from the next three lists:

      .. code-block:: bash

         drevalpy list sample-featurizers
         drevalpy list drug-featurizers
         drevalpy list predictors

      The ``predictors`` table, for example, begins like this and ends with the line
      ``30 entries``:

      .. code-block:: text

         Predictors
          Name                Description                                      Tags
          ─────────────────────────────────────────────────────────────────────────────
          naiveMeanEffects    Predict mean plus sample and drug effects.       baseline
          naiveDrugMean       Predict per-drug mean response with global fallback.  baseline
          ...
          lightgbm            LightGBM regressor on concatenated dense features.    -
          elasticNet          Elastic Net regression on concatenated dense features. -
          ...

   .. tab-item:: Python
      :sync: python

      The registries are in ``drevalpy.registry``. Each has ``list()`` for the names, ``table()``
      for a pandas data frame, and ``metadata(name)`` for one entry. The zoo is in
      ``drevalpy.models.zoo``.

      .. code-block:: python

         from drevalpy import registry
         from drevalpy.models import zoo

         print(len(zoo.list_zoo_names()), zoo.list_zoo_names()[:4])
         print(registry.predictor.list()[:4])
         print(registry.sample_featurizer.list()[:4])
         print(registry.drug_featurizer.list()[:4])

      The calls print:

      .. code-block:: text

         33 ['AdaBoostDecisionTree', 'DIPK', 'DrugGNN', 'ElasticNet']
         ['naiveMeanEffects', 'naiveDrugMean', 'naiveSampleMean', 'naiveMean']
         ['constant', 'identity', 'concatFeaturizers', 'dipkGeneExpression']
         ['constant', 'identity', 'concatFeaturizers', 'bpePharmaformer']

Naive baselines appear twice: as zoo names such as ``NaiveDrugMeanPredictor``, which you pass to
``run``, and as the ``naive*`` predictors, such as ``naiveDrugMean``, which are the components
behind them. The zoo name of the ``naiveMean`` predictor is ``NaivePredictor``.

List split modes and plots
--------------------------

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy list splitters
         drevalpy list visualizations

      The splitters table has five rows. ``LCO`` is listed as an alias of ``LSO``, the former
      name of that mode:

      .. code-block:: text

         Splitters
          Mode   Description                                                Validation   Aliases
          ──────────────────────────────────────────────────────────────────────────────────────
          LDO    Leave-Drug-Out: test folds contain unseen drugs            LDO          -
          LPO    Leave-Pair-Out: groups by (sample, drug) pairs             LPO          -
          LSO    Leave-Sample-Out: test folds contain unseen samples        LSO          LCO
          LStO   Leave-Study-Out: test folds contain one unseen study       LStO         -
          LTO    Leave-Tissue-Out: test folds contain unseen tissue types   LTO          -

      The visualizations table has six rows. Its last two columns show which result a plot
      draws from and what it needs. A plot that needs several folds or several models is left
      out of the report when the experiment has fewer; see :doc:`/how-to/make_report`.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         print(registry.splitter.list())
         print(registry.visualization.list())

      The calls print:

      .. code-block:: text

         ['LDO', 'LPO', 'LSO', 'LStO', 'LTO']
         ['comparison_scatter', 'critical_difference', 'heatmap', 'leaderboard', 'regression_scatter', 'violin']

List datasets and plugins
-------------------------

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy list datasets
         drevalpy list plugins

      The datasets table has a row per name, with the source it comes from and whether it is
      built in or registered by you:

      .. code-block:: text

         Datasets
          Name        Source   File             Origin
          ──────────────────────────────────────────────
          BeatAML2    orakl    BeatAML2.h5mu    built-in
          CTRPv1      orakl    CTRPv1.h5mu      built-in
          ...
          TOYv1       orakl    TOYv1.h5mu       built-in

          7 entries

      Without installed plugins, ``drevalpy list plugins`` says that no package declares a
      ``drevalpy.plugins`` entry point. With plugins, it lists each loaded one, and the ones
      that failed to load with the reason.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         print(registry.dataset.list())
         print(registry.get_loaded_plugins(), registry.get_failed_plugins())

      The calls print the seven names and then two empty mappings, when no plugin is installed:

      .. code-block:: text

         ['BeatAML2', 'CTRPv1', 'CTRPv2', 'GDSC1', 'GDSC2', 'PDX_Bruna', 'TOYv1']
         {} {}

Look at one entry
-----------------

Pass a name to see the full metadata of one entry: the components of a zoo preset, the aliases
of a split mode, the URL of a dataset.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy list zoo-presets ElasticNet

      .. code-block:: text

         ElasticNet
          Field               Value
          ────────────────────────────────────────────────────────
          Recipe              scaledGeneExpression:fingerprints:elasticNet
          Scope               multi_drug
          Prediction mode     regression
          Sample featurizer   scaledGeneExpression
          Drug featurizer     fingerprints
          Predictor           elasticNet

      An unknown name prints the available names and exits with status 1, so a script can
      detect it.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         print(zoo.get_zoo_config("ElasticNet").predictor.name)
         print(registry.predictor.metadata("elasticNet")["description"])
         print(registry.splitter.metadata("LSO"))

      The first two calls print:

      .. code-block:: text

         elasticNet
         Elastic Net regression on concatenated dense features.

The same works for every subcommand, for example ``drevalpy list datasets GDSC1``.
