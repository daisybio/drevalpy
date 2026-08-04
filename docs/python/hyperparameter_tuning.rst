Hyperparameter tuning
=====================

If you are reading this, we assume you are already familiar with this
concept:

- :doc:`/concepts/from_components_to_models`

Component predictors (and tunable featurizers such as ``pca`` or
``landmarkGenes``) own:

- ``get_default_hyperparameters()`` for the public flat dict used when
  constructing ``Model()`` / ``Model(hyperparameters)``
- ``get_hyperparameter_space()`` for structured Ray + Optuna search

The meaning of dotted keys
(``predictor.elasticNet.alpha``,
``cell_line_featurizer.pca[expression].n_components``, …) is defined in
:doc:`/concepts/from_components_to_models`. Featurizer keys use the
**qualified recipe selector** (including the view bracket when present).
Indexed forms such as ``pca.0`` are rejected. Flat constructor dicts still use
local names without a selector (e.g. ``n_components``).

Public API
----------

- ``DRPModel.get_hyperparameter_set()`` returns a **single** default
  configuration (``[get_default_hyperparameters()]``), not a full Cartesian
  grid.
- ``DRPModel.get_structured_hyperparameter_space()`` exposes the tunable
  search space with dotted keys.
- Experiment tuning uses ``hyperparameter_tuning=True`` with
  ``hpo_num_samples``, ``hpo_random_state``, and ``hpo_resources_per_trial``
  (Ray Tune + Optuna).

When ``hyperparameter_tuning=False``, experiments use defaults only. This is
**not** a debug mode and does not iterate legacy grid entries.

Ray vs Optuna
-------------

These are not two alternate backends. They play different roles in one stack:

- **Ray Tune** runs and schedules trials (parallelism,
  ``hpo_resources_per_trial``, trial storage).
- **Optuna** (via Ray’s ``OptunaSearch``) chooses which hyperparameter values
  to try next and optimizes the chosen metric.

Without Ray installed, ``hyperparameter_tuning=True`` fails at import time.
Set ``hyperparameter_tuning=False`` for defaults-only runs, or install on a
platform that has Ray wheels (see :doc:`/getting_started/installation`).

Fix the architecture before tuning
----------------------------------

Inputs (which omics / drug representation) are part of the model architecture,
not HPO knobs. Choose a zoo preset, ``ModelConfig``, or recipe first — see
:doc:`model_inputs` and :doc:`/concepts/from_components_to_models` — then tune
predictor / featurizer hyperparameters on that fixed stack.

Running HPO from Python
-----------------------

Use the root experiment (Ray Tune + Optuna) or call ``hpam_tune`` /
``tune_fold`` with a model **class**:

.. code-block:: python

   from drevalpy.experiment import drug_response_experiment
   from drevalpy.models import construct_model

   ElasticNet = construct_model("ElasticNet")

   drug_response_experiment(
       models=[ElasticNet],
       response_data=response_data,
       hyperparameter_tuning=True,
       hpo_num_samples=16,
       hpo_random_state=42,
   )

.. code-block:: python

   from drevalpy.components.tuning import hpam_tune

   best = hpam_tune(
       model_class=ElasticNet,
       train_dataset=train,
       validation_dataset=val,
       early_stopping_dataset=None,
   )

Inspect the structured space:

.. code-block:: python

   space = ElasticNet.get_structured_hyperparameter_space()

Component search spaces look like:

.. code-block:: python

   {"alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0}}

Defaults for ``Model()`` still come from the classmethod
``get_default_hyperparameters()``.

Migration notes
---------------

``hpam_tune`` naming
~~~~~~~~~~~~~~~~~~~~

Before 1.6.0, sequential grid search lived in ``drevalpy.experiment.hpam_tune``,
and Ray-based search was exposed as ``hpam_tune_raytune`` /
``hpam_tune_ray_optuna``. Those names are gone. ``hpam_tune`` now means only
the Ray Tune + OptunaSearch path
(``drevalpy.components.tuning.hpam_tune``, also re-exported from
``drevalpy.experiment``). Use ``hyperparameter_tuning=True`` in experiments, or
``hyperparameter_tuning=False`` for defaults.

YAML grids
~~~~~~~~~~

Before 1.6.0, baseline tuning used YAML grids such as:

.. code-block:: yaml

   ElasticNet:
     alpha:
       - 1
       - 0.1
       - 10

Those grids are gone. Translate ranges into component search spaces and enable
Ray search with ``hyperparameter_tuning=True``.
``get_hyperparameter_set()`` now returns one default dict only.

``multiprocessing`` and views
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before 1.6.0, ``multiprocessing=True`` selected a parallel HPO path. It now
only emits a warning and does **not** control Ray/Optuna tuning. Prefer
``hyperparameter_tuning=True`` and ``hpo_num_samples``.

Deprecated flat ``cell_line_views`` / ``drug_views`` keys are documented under
:doc:`architecture` migration notes; prefer recipes or zoo blocks
(:doc:`model_inputs`).
