Migrating to the v2 architecture
=================================

The v2 architecture moves callers to ``ModelSpec`` and ``DRPModel`` while
keeping split and evaluation protocols explicit. Re-run and re-save fitted
models after upgrading. Checkpoints from the previous architecture are not
adapted: loading an old checkpoint fails and the model must be trained again.

Update model construction
-------------------------

Use a specification and the model wrapper:

.. code-block:: python

   from drevalpy.models import DRPModel, ModelSpec

   spec = ModelSpec.from_name("ElasticNet")
   model = DRPModel(spec, hyperparameters=None)
   model.train(dataset, scope)
   prediction = model.predict(dataset, scope)

For a configured model, pass ``ModelSpec(name, config)`` with a ``ModelConfig``
such as the result of ``from_spec``. The model owns a clone of that
configuration. There are no generated model classes in the new surface.
Callers should construct a ``ModelSpec`` and pass it to ``DRPModel``.

Persist scaling with the model
------------------------------

Target scaling is fitted with the model and stored in its checkpoint. Loading
therefore returns predictions in the original response units:

Pass an unfitted transformer through ``DRPModel(...,
response_transformation=...)`` when a custom transformation is needed. The
pipeline now owns fitting and inverse transformation; the former standalone
response-fitting helper is no longer part of the public API.

.. code-block:: python

   model.train(dataset, scope, model_checkpoint_dir="checkpoints")
   model.save("checkpoints/model")
   restored = DRPModel.load("checkpoints/model")
   prediction = restored.predict(dataset, scope)  # original response units

If the path contains an old checkpoint, retrain and save a new v3 checkpoint.
Do not combine predictions produced before and after the migration in one
result table without recording the model version and rerunning the affected
folds.

Use the experiment helpers
--------------------------

The multi-model and one-fold entry points accept ``ModelSpec`` values:

.. code-block:: python

   from drevalpy import run, single

   results = run(models=[spec], dataset=dataset, split_mode="LSO", hyperparameter_tuning=False)
   one_fold = single(spec, dataset, fold, hyperparameter_tuning=False)

Experiments create immutable fit and execution scopes. A model that supports
early stopping may receive ``early_stopping_scope`` in ``train``. For those models,
with or without HPO, the protocol retains the 25% E holdout derived from validation
data: candidates fit on A, select on V, and use E for stopping. The final fit uses
A union V, while E and test remain held out. A, E and V are disjoint. Tiny
validation sets retain the existing optional-stopping fallback; components that
require a usable holdout fail clearly. Models without early stopping fit on the
original training and validation selections together.

Featurizer and predictor extensions
------------------------------------

Implement extensions around ``FeaturizerResult(entity_ids, blocks,
projection)``. The projection is a tuple of block names. A single block keeps
its original payload; multiple blocks are converted to a matrix only when all
are numeric and aligned. ``ModelInputBatch`` references these results for the
predictor. This supports graph and object blocks without forcing them through
a numeric concatenation step.

The internal pipeline owns ``UNTRAINED``, ``EMPTY`` and ``FITTED``. Do not
persist an empty fit as a reusable checkpoint. An empty fit returns NaN for
requested predictions and should be handled as a run result.

Metrics and correctness checks
------------------------------

For one pair, the supported error metrics are MSE, RMSE and MAE. R2 and
correlation metrics require at least two observations; configure evaluation to
skip them or report them as unavailable for a one-pair result. Keep all eight
numerical parity gates enabled when validating the migration. Matching one
pair on an error metric is not evidence that a correlation or R2 implementation
is valid.

Migration checklist
-------------------

1. Replace generated model construction with ``ModelSpec.from_name`` or
   ``ModelSpec(name, config)``.
2. Construct ``DRPModel`` and call ``train``/``predict`` with explicit scopes.
3. Refit models and save v3 checkpoints so target scaling is persisted.
4. Follow the rerun guidance below for existing predictions and reports.
5. Verify one-pair metrics use MSE, RMSE or MAE, and that R2/correlation
   metrics require two observations.
6. Run all eight numerical parity gates and compare results using the full
   run identity.

Which experiments need to run again?
------------------------------------

* Retrain any model whose pre-v3 checkpoint you need to load. Its target
  transformation cannot safely be reconstructed from the old archive.
* Update direct prediction consumers that expected transformed target units or
  inverted predictions themselves, then regenerate their predictions and metrics.
  Experiment and HPO predictions were already reported in original units and
  must remain numerically equivalent under the same valid protocol.
* Recompute metrics for scopes with exactly one valid pair: MSE, RMSE and MAE
  are now defined. R2 and correlations remain NaN.
* Rerun experiments produced by the former leaking holdout protocol. Those
  scores are not a valid performance floor for leakage-free training.
* Regenerate execution records when complete configuration and scope provenance
  is required. Existing leakage-free results with aligned original-unit
  predictions and at least two valid pairs do not require retraining solely
  because of this structural change. Historical diagnostic reports must use
  ordinary reference runs and preserve each variant's identity.
