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
   constructor or ``config.from_yaml(...)``, then pass that object as
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

         from drevalpy.models import config, construct_model

         cfg = config.from_yaml("my_zoo/custom_rf.yaml")
         MyRF = construct_model("MyRF", cfg)

   .. tab-item:: ModelConfig

      .. code-block:: python

         from drevalpy.models import config, construct_model

         cfg = config.ModelConfig(
             cell_line_featurizer=config.CellLineFeaturizerConfig(
                 name="scaledGeneExpression"
             ),
             drug_featurizer=config.DrugFeaturizerConfig(name="fingerprints"),
             predictor=config.PredictorConfig(name="randomForest"),
         )
         MyRF = construct_model("MyRF", cfg)

   .. tab-item:: ModelConfig + hyperparameter space

      Set ``hyperparameter_space`` on a component to **replace** its built-in
      search space (see :doc:`/concepts/from_components_to_models`). Recipe
      strings cannot express this; use YAML or ``ModelConfig``.

      .. code-block:: python

         from drevalpy.models import config, construct_model

         cfg = config.ModelConfig(
             cell_line_featurizer=config.CellLineFeaturizerConfig(
                 name="scaledGeneExpression"
             ),
             drug_featurizer=config.DrugFeaturizerConfig(name="fingerprints"),
             predictor=config.PredictorConfig(
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
         MyEN = construct_model("MyElasticNet", cfg)

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

Aliases such as ``methylation_n_components`` remain accepted on input.
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
