Models
======

The public model API has two pieces: ``ModelSpec`` describes a named model and
its configuration, and ``DRPModel`` owns fitting, prediction, response
transformation and persistence. The component pipeline behind the wrapper is
an implementation boundary.

Selecting and configuring a model
---------------------------------

Use a zoo name for a built-in model:

.. code-block:: python

   from drevalpy.models import ModelSpec

   spec = ModelSpec.from_name("ElasticNet")

For a custom recipe, pass the model name and a ``ModelConfig``:

.. code-block:: python

   from drevalpy.models.config import from_spec

   spec = ModelSpec(
       "MyElasticNet",
       from_spec("scaledGeneExpression:fingerprints:elasticNet"),
   )

The specification owns an immutable configuration snapshot. Its name identifies
this model in an experiment; ``ModelConfig.model_id`` labels the recipe and
need not be unique. Give each specification in an experiment a unique name.

Constructing and fitting
------------------------

.. code-block:: python

   from drevalpy.models import DRPModel

   model = DRPModel(spec, hyperparameters=None)
   model.train(dataset, scope)
   prediction = model.predict(dataset, scope)

Pass ``hyperparameters`` when a run needs an override. A response
transformation can be selected with the keyword-only
``response_transformation`` argument. ``train`` also accepts an optional
``early_stopping_scope`` and ``model_checkpoint_dir`` (default
``"checkpoints"``). The model keeps the fitted target transformation and
returns predictions in the original response units.

For example, standardize targets during fitting and keep public predictions in
the dataset's original units:

.. code-block:: python

   from sklearn.preprocessing import StandardScaler

   model = DRPModel(spec, response_transformation=StandardScaler())
   model.train(dataset, scope)
   prediction = model.predict(dataset, scope)

The supplied transformer is a prototype. Each training attempt fits a new clone
on observed training responses, before feature eligibility filtering, and never
fits it on stopping or test responses. Custom sklearn-compatible transformers
must implement ``fit``, ``transform`` and ``inverse_transform`` for one column.

Feature construction produces ``FeaturizerResult`` values and a
``ModelInputBatch`` for the predictor. Extensions are covered in
:doc:`extensions`.

Persistence
-----------

Save a fitted model and restore it with the same public wrapper:

.. code-block:: python

   model.train(dataset, scope, model_checkpoint_dir="checkpoints")
   model.save("checkpoints/elastic_net")

   restored = DRPModel.load("checkpoints/elastic_net")
   prediction = restored.predict(dataset, scope)  # original units

Only fitted models can be saved. ``DRPModel.load`` restores v3 checkpoints,
including the target scaler. Older checkpoint formats are rejected and must be
retrained so that response units and feature contracts are unambiguous.

Lifecycle and scopes
--------------------

The internal pipeline has three states: ``UNTRAINED``, ``EMPTY`` and
``FITTED``. An empty fit means no usable pairs survived filtering; prediction
for requested pairs is NaN and the empty state is not a reusable checkpoint.

Experiment helpers create immutable fit scopes. The ordinary protocol uses A
for training, E for early stopping, V for remaining validation pairs and test
for held-out evaluation. E and V are disjoint, and test is disjoint from all
three. See :doc:`experiments` and :doc:`/project/v2_architecture` for the
ownership and execution model.
