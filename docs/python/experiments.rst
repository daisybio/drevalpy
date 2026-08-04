Experiments
===========

If you are reading this, we assume you are already familiar with these
concepts:

- :doc:`/concepts/evaluation`
- :doc:`/concepts/from_components_to_models`

:func:`~drevalpy.experiment.drug_response_experiment` runs nested
cross-validation, optional hyperparameter tuning, baselines, and optional
randomization or robustness tests. Results are written under ``path_out`` /
``run_id`` / dataset / split label.

For day-to-day benchmarking, prefer this runner over a hand-rolled
train/predict loop on individual models (see :doc:`models` for the
low-level lifecycle).

Minimal call
------------

.. code-block:: python

   from drevalpy.datasets.loader import load_dataset
   from drevalpy.experiment import drug_response_experiment
   from drevalpy.models import construct_model

   response_data = load_dataset("TOYv1", path_data="data")
   ElasticNet = construct_model("ElasticNet")

   drug_response_experiment(
       models=[ElasticNet],
       response_data=response_data,
       run_id="en_toy",
       test_mode="LCO",
       n_cv_splits=5,
       path_data="data",
       path_out="results/",
       hyperparameter_tuning=False,
   )

Pass model **classes** from :func:`~drevalpy.models.construct_model`, not
instances. ``NaiveMeanEffectsPredictor`` is always included among baselines
when missing — it is required for normalized metrics.

Common options
--------------

- ``test_mode``: ``LPO``, ``LCO``, ``LTO``, or ``LDO`` (see
  :doc:`/concepts/evaluation`).
- ``baselines``: extra baseline classes; randomization/robustness apply only to
  ``models``.
- ``hyperparameter_tuning``: ``True`` tunes over each model's structured
  search space; ``False`` uses each model's
  ``get_default_hyperparameters()`` only.
- ``hpo_num_samples``, ``hpo_random_state``, ``hpo_resources_per_trial``:
  control the search when tuning is on.
- ``randomization_mode`` / ``randomization_type`` / ``n_trials_robustness``:
  stress tests (see :doc:`/concepts/evaluation`).
- ``cross_study_datasets``: other ``DrugResponseDataset`` instances for
  transfer evaluation.
- ``custom_splitter`` / ``custom_split_name``: replace built-in
  ``split_dataset`` with an external split creator.
- ``final_model_on_full_data``: optionally fit and persist a production model
  after CV.
- ``wandb_project``: enable Weights & Biases logging when set.

Example with tuning enabled:

.. code-block:: python

   drug_response_experiment(
       models=[ElasticNet],
       response_data=response_data,
       run_id="en_toy_hpo",
       test_mode="LCO",
       hyperparameter_tuning=True,
       hpo_num_samples=16,
       hpo_random_state=42,
       hpam_optimization_metric="RMSE",
   )

See :doc:`visualization` for reports over the written predictions.

Hyperparameter tuning
---------------------

Component predictors (and tunable featurizers such as ``pca`` or
``landmarkGenes``) own:

- ``get_default_hyperparameters()`` for the public hyperparameter mapping used when
  constructing ``Model()`` / ``Model(hyperparameters)``
- ``get_hyperparameter_space()`` for structured Ray + Optuna search

The meaning of qualified keys
(``predictor.elasticNet.alpha``,
``cell_line_featurizer.pca[expression].n_components``, …) is defined in
:doc:`/concepts/from_components_to_models`. Featurizer keys use the
**qualified recipe selector** (including the view bracket when present).
Indexed forms such as ``pca.0`` are rejected.

Constructor mappings use local names when they are unambiguous. When a local
name collides, pass qualified keys instead. ``hpam_tune`` and saved best-result
JSON use the same collision-aware public mapping: compact short keys when
possible, qualified keys when required. See :doc:`models` for construction-time
overrides.

Public API
~~~~~~~~~~

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
~~~~~~~~~~~~~

These are not two alternate backends. They play different roles in one stack:

- **Ray Tune** runs and schedules trials (parallelism,
  ``hpo_resources_per_trial``, trial storage).
- **Optuna** (via Ray's ``OptunaSearch``) chooses which hyperparameter values
  to try next and optimizes the chosen metric.

Without Ray installed, ``hyperparameter_tuning=True`` fails at import time.
Set ``hyperparameter_tuning=False`` for defaults-only runs, or install on a
platform that has Ray wheels (see :doc:`/getting_started/installation`).

Fix the architecture before tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs (which omics / drug representation) are part of the model architecture,
not HPO knobs. Choose a zoo preset, ``ModelConfig``, or recipe first — see
:doc:`datasets` and :doc:`/concepts/from_components_to_models` — then tune
predictor / featurizer hyperparameters on that fixed stack.

Running HPO from Python
~~~~~~~~~~~~~~~~~~~~~~~

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

HPO migration notes
~~~~~~~~~~~~~~~~~~~

``hpam_tune`` naming
^^^^^^^^^^^^^^^^^^^^

Before 1.6.0, sequential grid search lived in ``drevalpy.experiment.hpam_tune``,
and Ray-based search was exposed as ``hpam_tune_raytune`` /
``hpam_tune_ray_optuna``. Those names are gone. ``hpam_tune`` now means only
the Ray Tune + OptunaSearch path
(``drevalpy.components.tuning.hpam_tune``, also re-exported from
``drevalpy.experiment``). Use ``hyperparameter_tuning=True`` in experiments, or
``hyperparameter_tuning=False`` for defaults.

YAML grids
^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before 1.6.0, ``multiprocessing=True`` selected a parallel HPO path. It now
only emits a warning and does **not** control Ray/Optuna tuning. Prefer
``hyperparameter_tuning=True`` and ``hpo_num_samples``.

Deprecated flat ``cell_line_views`` / ``drug_views`` keys are documented under
:doc:`models` migration notes; prefer recipes or zoo blocks
(:doc:`datasets`).
