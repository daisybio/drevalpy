v2 architecture
================

v2 presents model training as a small, explicit public API. A caller chooses a
model specification and a dataset; the library owns feature construction,
fitting, response scaling, persistence and prediction. The public surface is
deliberately smaller than the collection of implementation components behind
it.

Public model API
----------------

``ModelSpec`` identifies a named model and its configuration. Built-in models
are selected by name and custom configurations use ``from_spec``:

.. code-block:: python

   from drevalpy.models import DRPModel, ModelSpec

   elastic_net = ModelSpec.from_name("ElasticNet")
   model = DRPModel(elastic_net, hyperparameters=None)
   model.train(dataset, scope)
   predictions = model.predict(dataset, scope)

   from drevalpy.models.config import from_spec

   custom = ModelSpec(
       "ElasticNet",
       from_spec("scaledGeneExpression:fingerprints:elasticNet"),
   )

The constructor accepts ``DRPModel(spec, hyperparameters=None, *,
response_transformation=None)``. A response transformation is part of the
model state, so the same transformation is applied after loading a model and
predictions remain in the original response units. Models own the clones of
their input configuration and fitted components; a caller can safely reuse a
dataset or specification for another run.

Training and prediction take a dataset and an explicit scope. ``early_stopping_scope``
is optional, and ``model_checkpoint_dir`` defaults to ``"checkpoints"``:

.. code-block:: python

   model.train(
       dataset,
       scope,
       early_stopping_scope=validation_scope,
       model_checkpoint_dir="checkpoints",
   )
   predictions = model.predict(dataset, scope)

``DRPModel.load(path)`` restores a v3 checkpoint, including the target scaler
and fitted state. Older checkpoint formats are rejected with a request to
retrain; silently adapting a fitted object would make its response units and
feature contract ambiguous.

Save a fitted model explicitly before loading it in another process:

.. code-block:: python

   model.save("checkpoints/elastic_net")
   restored = DRPModel.load("checkpoints/elastic_net")

Experiment API
--------------

The experiment helpers use the same model API for one or many models:

.. code-block:: python

   from drevalpy import run, single

   results = run(models=[elastic_net], dataset=dataset, split_mode="LCO", hyperparameter_tuning=False)
   fold_result = single(elastic_net, dataset, fold, hyperparameter_tuning=False)

``run`` owns fold execution and result identity. ``single`` is the focused
entry point for one model and one fold, useful for inspection and debugging.
Neither helper requires generated model classes: both accept ``ModelSpec``
values and leave feature construction and fitting to the model pipeline.

Ownership boundaries
--------------------

Experiments own protocol decisions; the model pipeline owns fitted behavior. A new feature should be added at the narrowest owner
that has the required information.

=======================  ================================================
Owner                    Responsibility
=======================  ================================================
``ModelSpec``             Name and immutable configuration of a model.
``DRPModel``              Model identity, cloned state, target scaling and
                          persistence of a fitted model.
Internal pipeline         Featurization, component fitting, state transitions
                          (``UNTRAINED``, ``EMPTY``, ``FITTED``), and assembly
                          of model inputs.
``FitScopes``             Immutable selections for tuning, stopping, final
                          training, validation and test for one fit.
``ExecutionSpec``         Immutable choice of ordinary, randomization and
                          robustness execution settings.
``RunIdentity``           Stable identity for dataset, fold, model and
                          execution variant in results and reports.
``FeaturizerResult``      Entity IDs, named feature blocks and their projection
                          order.
``ModelInputBatch``        References the results consumed by a predictor.
Experiment helpers        Fold iteration, split protocol and result reporting.
=======================  ================================================

The internal pipeline replaces the old ``component_stack`` as the orchestration
boundary. Its three states are explicit: an untrained pipeline cannot predict;
an empty pipeline records that no usable pairs survived filtering and returns
NaN predictions for requested pairs; and a fitted pipeline is the only state
that can be saved and loaded.

Architecture at a glance
------------------------

.. code-block:: text

   ModelSpec + Dataset + original SplitMasks
                     |
              experiment scopes / HPO
                     |
           DRPModel(private fitted pipeline)
              |                    |
       target transform      featurizers
              |                    |
       transformed targets + ModelInputBatch
                     |
                 predictor
                     |
           inverse target transform
                     |
       original-unit, scope-aligned predictions
                     |
           shared evaluation -> results

Python argument-binding errors leave an existing fit intact. Once ``train``
is entered, a failed attempt leaves the pipeline ``UNTRAINED``; an empty fit
leaves it ``EMPTY``. Neither state can expose or save the previous fitted model.

Feature data flow
-----------------

``FeaturizerResult`` is canonical and contains ``entity_ids``, ``blocks`` and
``projection`` (a tuple of block names). A matrix is derived from blocks only
when a numeric matrix is meaningful:

* an empty projection produces a matrix with shape ``(len(entity_ids), 0)``;
* one block is returned unchanged, including graph or object payloads;
* multiple blocks are concatenated only when all blocks are numeric and
  entity-aligned; graph and object blocks remain addressable through the
  block mapping.

``ModelInputBatch`` references the results needed by a predictor instead of
copying or rebuilding them. This lets predictors consume structured blocks
while models that need a matrix obtain the deterministic projection.

Fit and prediction traces
-------------------------

A regular fit follows this sequence:

.. code-block:: text

   run(models=[spec], dataset=dataset, split_mode="LCO")
     -> RunIdentity(fold, model, ordinary execution)
     -> immutable ExecutionSpec(FitScopes)
     -> DRPModel clones config
     -> model.train(dataset, final_training, early_stopping_scope=E)
          -> invalidate old state; create fresh components once
          -> validate scopes; extract observed training targets
          -> clone and fit target transformer on those targets
          -> filter unusable input rows
          -> fit featurizers on training entities only
          -> compute FeaturizerResult(s) once; build ModelInputBatch
          -> filter invalid feature rows and stopping targets
          -> predictor fits transformed targets, stops using E
          -> FITTED model
     -> model.predict(dataset, test)

For HPO and early stopping, A is the original training scope, E is the 25%
stopping holdout carved from validation, and V is the remaining validation
scope. A, E, V and test are disjoint. Candidate models train on A, select on
V, and may stop using E; the final fitted model trains on A union V while E
remains reserved for stopping. The model receives these scopes and never
partitions them. An explicit ``early_stopping_scope`` selects E without
changing the identity of the ordinary run.

Prediction loads or receives a fitted model, constructs the requested feature
results, and reverses target scaling before returning values:

.. code-block:: text

   DRPModel.load(path)
     -> restore target scaler and fitted predictor
     -> model.predict(dataset, scope)
          -> check state before reading features
          -> EMPTY returns a NaN per requested position; UNTRAINED raises
          -> compute features once, retaining fitted output schemas
          -> predictor outputs transformed units for eligible pairs
          -> restore invalid positions as NaN; inverse target transformation
          -> prediction in original response units and full scope order

Randomization and robustness executions retain the same fold identity but get
their own immutable ``ExecutionSpec`` and ``RunIdentity``. Ordinary metrics
and reports select ordinary executions by default, so diagnostic variants do
not overwrite or silently enter the main leaderboard.

Extension guidance
------------------

Most users need only a named ``ModelSpec``. Plugin authors can provide a
registered model configuration and its feature blocks; they should preserve
the ``FeaturizerResult`` and ``ModelInputBatch`` contracts and let the internal
pipeline own fitting state. Advanced plugins may implement structured or
graph blocks, custom response transformations, or early stopping, but should
not recreate the public training wrapper.
