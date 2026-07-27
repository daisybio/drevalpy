Experiments
===========

``drug_response_experiment`` runs nested cross-validation, optional
hyperparameter tuning, baselines, and optional randomization or robustness
tests. Results are written under ``path_out`` / ``run_id`` / dataset / split
label.

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

Pass model **classes** (from ``construct_model`` or named facades), not
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

See :doc:`hyperparameter_tuning` for search spaces, dotted keys, and the
Ray/Optuna backend, and :doc:`visualization` for reports over the written
predictions.

Backward compatibility
----------------------

multiprocessing
~~~~~~~~~~~~~~~

Before 1.6.0, ``multiprocessing=True`` was used as a parallel HPO switch. It
now only emits a warning and does **not** control hyperparameter tuning. This
remains available for backward compatibility, but is deprecated and may be
removed in a future release. Prefer ``hyperparameter_tuning=True`` with
``hpo_num_samples``.

get_hyperparameter_set
~~~~~~~~~~~~~~~~~~~~~~

Before 1.6.0, ``DRPModel.get_hyperparameter_set()`` returned a Cartesian grid
from YAML. It now returns a **single** default configuration
(``[get_default_hyperparameters()]``). Callers that looped the old grid should
switch to ``hyperparameter_tuning=True`` or
``get_structured_hyperparameter_space()``.

hyperparameter_tuning=False
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before 1.6.0, disabling tuning could still walk legacy grid entries. Now
``hyperparameter_tuning=False`` means **defaults only** — it is not a debug
mode and does not iterate an old ParameterGrid.
