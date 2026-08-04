Models
======

If you are reading this, we assume you are already familiar with these
concepts:

- :doc:`/concepts/component_catalog`
- :doc:`/concepts/from_components_to_models`
- :doc:`/concepts/model_zoo`

Every runnable model in DrEvalPy is a thin ``DRPModel`` subclass produced by
:func:`~drevalpy.models.construct_model`. You never hand-write that subclass:
you declare a ``ModelConfig`` (or something that becomes one), resolve a
**class**, then construct a fresh **instance**. For Python orchestration
details see :doc:`architecture`.

From declaration to instance
----------------------------

A ``ModelConfig`` is only a description of the featurizer/predictor stack.
``construct_model`` turns that description into a class; calling the class
produces a runnable object. Three kinds of input all converge on the same
``ModelConfig`` before resolution:

.. mermaid::

   flowchart TD
      subgraph specInputs ["Declare a stack"]
         zooPreset["Zoo preset name"]
         recipeString["Recipe string"]
         yamlOrDict["YAML or dict"]
      end
      modelConfig["ModelConfig"]
      constructModel["construct_model(name, spec)"]
      drpSubclass["DRPModel subclass"]
      instance["ModelClass(hyperparameters)"]

      zooPreset --> modelConfig
      recipeString --> modelConfig
      yamlOrDict --> modelConfig
      modelConfig --> constructModel
      constructModel --> drpSubclass
      drpSubclass --> instance

Follow the graph left to right. The tabs below show equivalent ways to declare
a stack, resolve a ``DRPModel`` subclass with ``construct_model``, and
construct an instance:

.. tab-set::

   .. tab-item:: Zoo

      .. code-block:: python

         from drevalpy.models import construct_model

         ElasticNet = construct_model("ElasticNet")
         model = ElasticNet()
         model = ElasticNet({"alpha": 0.1})

      Discover names with ``list_zoo_names()`` (optionally filter by
      ``ModelScope``). Presets are listed in :doc:`/concepts/model_zoo`.

   .. tab-item:: Recipe string

      .. code-block:: python

         from drevalpy.models import construct_model

         MyRF = construct_model(
             "MyRF",
             "scaledGeneExpression:fingerprints:randomForest",
         )
         model = MyRF({"n_estimators": 200})

   .. tab-item:: YAML

      .. code-block:: yaml

         cell_line_featurizer: scaledGeneExpression
         drug_featurizer: fingerprints
         predictor: randomForest

      .. code-block:: python

         from drevalpy.models import construct_model
         from drevalpy.models.config import ModelConfig

         config = ModelConfig.from_yaml("my_zoo/custom_rf.yaml")
         MyRF = construct_model("MyRF", config)
         model = MyRF({"n_estimators": 200})

   .. tab-item:: ModelConfig

      .. code-block:: python

         from drevalpy.models import construct_model
         from drevalpy.models.config import (
             CellLineFeaturizerConfig,
             DrugFeaturizerConfig,
             ModelConfig,
             PredictorConfig,
         )

         config = ModelConfig(
             cell_line_featurizer=CellLineFeaturizerConfig(
                 name="scaledGeneExpression"
             ),
             drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
             predictor=PredictorConfig(name="randomForest"),
         )
         MyRF = construct_model("MyRF", config)
         model = MyRF({"n_estimators": 200})

   .. tab-item:: ModelConfig + hyperparameter space

      Set ``hyperparameter_space`` on a component to **replace** its built-in
      search space (see :doc:`/concepts/from_components_to_models`). Recipe
      strings cannot express this; use YAML or ``ModelConfig``.

      .. code-block:: python

         from drevalpy.models import construct_model
         from drevalpy.models.config import (
             CellLineFeaturizerConfig,
             DrugFeaturizerConfig,
             ModelConfig,
             PredictorConfig,
         )

         config = ModelConfig(
             cell_line_featurizer=CellLineFeaturizerConfig(
                 name="scaledGeneExpression"
             ),
             drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
             predictor=PredictorConfig(
                 name="elasticNet",
                 hyperparameter_space={
                     "alpha": {
                         "type": "float",
                         "low": 1e-4,
                         "high": 10.0,
                         "log": True,
                         "default": 1.0,
                     },
                     "l1_ratio": {
                         "type": "float",
                         "low": 0.0,
                         "high": 1.0,
                         "default": 0.5,
                     },
                 },
             ),
         )
         MyEN = construct_model("MyElasticNet", config)
         model = MyEN()

Recipe grammar and YAML field names are documented in
:doc:`/concepts/from_components_to_models`. Applied featurizer examples (custom
CSV views) are in :doc:`model_inputs`; batch contracts and scope rules are in
:doc:`architecture`.

Lifecycle
---------

Each ``DRPModel`` subclass exposes:

1. ``ModelClass(hyperparameters=None)`` — construct with class defaults or flat
   overrides. Hyperparameters and view lists are immutable after construction;
   create a new instance to change configuration.
2. ``train(...)`` / ``predict(...)`` — fit and score on response + feature
   inputs (the experiment runner constructs a fresh instance per fold).
3. ``save(directory)`` / ``ModelClass.load(directory)`` / ``load_model(directory)`` —
   native ``model.joblib`` checkpoints (format ``drevalpy-model``; see
   :doc:`persistence`). Use ``load_model`` when you do not already have a
   class handle.

Predictors inside a ``ModelConfig`` receive static hyperparameters at
construction (``PredictorConfig.create_instance()``). Dimension-dependent
allocation happens privately during ``fit()``; there is no public
``Predictor.build``.

For day-to-day benchmarking, prefer
:func:`~drevalpy.experiment.drug_response_experiment` over a hand-rolled
train loop (:doc:`experiments`).

Migration notes
---------------

Before 1.6.0, ``MODEL_FACTORY``, ``MULTI_DRUG_MODEL_FACTORY``, and
``SINGLE_DRUG_MODEL_FACTORY`` were the usual lookup. They remain as **lazy,
built-in-only** compatibility views equivalent to ``construct_model(name)`` for
zoo preset names, but emit ``FutureWarning`` and may be removed in a future
release. Prefer ``construct_model``, ``ModelConfig.from_spec``, and
``list_zoo_names(scope=...)``. See :doc:`quickstart` for a short side-by-side
example.

Named root exports (``ElasticNetModel``, ``DIPKModel``, …) are removed. Use
``construct_model("ElasticNet")`` (or the zoo preset string) instead.

``ModelConfig.create_model()`` is removed. Use
``construct_model(name_or_recipe)()`` or ``construct_model(name, config)()``.

Deep imports such as ``drevalpy.models.DIPK.dipk`` or
``drevalpy.models.baselines.*`` no longer resolve. Resolve models with
``construct_model`` from ``drevalpy.models``.

Legacy checkpoint formats (including ``composed_model.joblib``) are not
loadable. Retrain and persist via ``model.save`` / ``ModelClass.load``
(``model.joblib``); see :doc:`persistence`.
