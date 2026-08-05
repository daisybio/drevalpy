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
**class**, then construct a fresh **instance**.

DrEvalPy has two cooperating layers: a **component stack** under
``drevalpy.components`` (featurizers, predictors, registries, tuning helpers)
and **public orchestration** under ``drevalpy.models`` (``ModelConfig``, zoo
YAML, and ``construct_model`` returning thin ``DRPModel`` subclasses). A
resolved instance materializes featurizer(s) and a predictor as an internal
component stack.

From declaration to instance
----------------------------

Constructing (custom) model classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~drevalpy.models.construct_model` is the only way of creating DRPModel classes.
There are two ways to call it:

#. With **one argument**, pass a zoo preset name (for example
   ``construct_model("ElasticNet")``). The name must exist in the model zoo.
#. With **two arguments**, pass a custom class name and a **spec** (for example
   ``construct_model("MyRF", spec)``). Use this form when the name is not a zoo
   preset. ``spec`` must be either a recipe string or a ``ModelConfig`` object —
   YAML paths are not accepted directly. Build a ``ModelConfig`` with the
   constructor or ``ModelConfig.from_yaml(...)``, then pass that object as
   ``spec``.

The tabs below show each call form:

.. tab-set::

   .. tab-item:: Zoo

      .. code-block:: python

         from drevalpy.models import construct_model

         ElasticNet = construct_model("ElasticNet")

      Discover names with ``list_zoo_names()`` (optionally filter by
      ``ModelScope``). Presets are listed in :doc:`/concepts/model_zoo`.

   .. tab-item:: Recipe string

      .. code-block:: python

         from drevalpy.models import construct_model

         MyRF = construct_model(
             "MyRF",
             "scaledGeneExpression:fingerprints:randomForest",
         )

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

Recipe grammar and YAML field names are documented in
:doc:`/concepts/from_components_to_models`. Applied featurizer examples with
custom CSV views are in :doc:`datasets`.

Instantiating models
~~~~~~~~~~~~~~~~~~~~

:func:`~drevalpy.models.construct_model` returns a ``DRPModel`` **subclass**, not a
runnable model. Call the class to create an instance:

.. code-block:: python

   from drevalpy.models import construct_model

   ElasticNet = construct_model("ElasticNet")

   model = ElasticNet()
   model = ElasticNet({"alpha": 0.1})

With no arguments, the constructor uses each component's default hyperparameters
(from the zoo preset, recipe, or ``ModelConfig``). Pass a **public hyperparameter
mapping** to override values at construction time.

Use **local** parameter names (``alpha``, ``n_components``, …) when the name is
unique in the stack — the usual case for zoo presets and simple recipes:

.. code-block:: python

   model = ElasticNet({"alpha": 0.1})

When the same local name exists on more than one component, DrEvalPy raises an
error and lists the accepted **qualified** keys. Target one slot explicitly:

.. code-block:: python

   model = MyModel(
       {
           "cell_line_featurizer.pca[expression].n_components": 32,
           "cell_line_featurizer.pca[proteomics].n_components": 16,
       }
   )

Legacy aliases such as ``methylation_n_components`` remain accepted on input.
Hyperparameters are fixed after construction; create a new instance to change
them.

Implementing predictors (``ModelInputBatch``, input interfaces, and
``FeatureFormat`` contracts) is covered in :doc:`custom_models`.

Training and persistence
------------------------

Each ``DRPModel`` subclass exposes:

1. ``train(...)`` / ``predict(...)`` — fit and score on response + feature
   inputs (the experiment runner constructs a fresh instance per fold).
2. ``save(path)`` / ``ModelClass.load(path)`` / ``load_model(path)`` —
   native ZIP checkpoints (format ``drevalpy-model``).

``save`` writes one deflated ZIP archive containing the resolved
``ModelConfig`` and fitted component state. If you already have a class handle,
you can use ``ModelClass.load`` for loading. Otherwise, use ``load_model``:
It first constructs the model class from the stored checkpoint and then loads the parameters.

.. code-block:: python

   from drevalpy.models import construct_model, load_model

   ElasticNet = construct_model("ElasticNet")
   model = ElasticNet()
   model.train(...)
   model.save("checkpoints/elastic_net")

   loaded = ElasticNet.load("checkpoints/elastic_net")
   loaded = load_model("checkpoints/elastic_net.zip")

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

Flat ``cell_line_views`` / ``drug_views`` in constructor or hpam YAML are
removed. Set ``cell_line_featurizer`` / ``drug_featurizer`` in zoo YAML or
a recipe string instead (see :doc:`datasets`).

Legacy checkpoint formats (including pickled ``.model`` attributes, standalone
scalers, and ``composed_model.joblib``) are not loadable. Retrain and persist
via ``model.save`` / ``ModelClass.load`` (``*.zip`` archives).
