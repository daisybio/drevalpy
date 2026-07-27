Hyperparameter tuning
=====================

DrEvalPy no longer uses YAML grid files for baseline tuning. Each component
predictor (and tunable featurizers such as ``pca`` or ``landmarkGenes``) owns:

- ``get_default_hyperparameters()`` for the public flat dict passed to
  ``build_model()``
- ``get_hyperparameter_space()`` for structured Ray + Optuna search

Public API
----------

- ``DRPModel.get_hyperparameter_set()`` returns a **single** default
  configuration (``[get_default_hyperparameters()]``), not a full Cartesian
  grid.
- ``DRPModel.get_structured_hyperparameter_space()`` exposes the tunable
  search space with dotted keys (``predictor.elasticNet.alpha``,
  ``featurizer.cell_line.pca.0.n_components``, …). For featurizers, the
  integer after the name is a **zero-based occurrence index** of that
  featurizer name in the composed stack (per registry). A single ``pca`` is
  always ``…pca.0.…``; with ``concatFeaturizers`` and several of the same name
  you get ``0``, ``1``, …. The index is **required** in structured dotted keys
  — keys without it are not applied. Flat ``build_model`` dicts still use
  names without an index (e.g. ``n_components``). See :doc:`architecture` for
  more examples.
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

Configuring inputs
------------------

Use one of:

- Zoo presets in ``drevalpy/models/zoo/*.yaml`` with explicit
  ``cell_line_featurizer`` / ``drug_featurizer`` blocks
- ``ModelConfig.from_spec()`` / ``ModelConfig.from_yaml()``
- Recipe strings such as ``raw[proteomics]:fingerprints:randomForest``

In a recipe, ``:`` separates cell-line featurizer, drug featurizer, and
predictor. Within a featurizer slot, ``+`` concatenates several featurizers
into ``concatFeaturizers`` (for example
``raw[expression]+pca[methylation]:fingerprints:xgboost``). Multi-view models
such as ``MultiViewXGBoost`` and ``MultiViewLightGBM`` are expressed this way
(or with equivalent zoo YAML blocks), not by special multi-view predictor
classes. See :doc:`architecture` for the full recipe grammar.

Running HPO from Python
-----------------------

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

Component search spaces look like:

.. code-block:: python

   {"alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0}}

Defaults for ``build_model()`` still come from the predictor's
``get_default_hyperparameters()`` implementation.

Backward compatibility
----------------------

Views as hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~

Before 1.6.0, ``cell_line_views`` / ``drug_views`` were treated as
hyperparameters. This remains available for backward compatibility, but is
deprecated and may be removed in a future release. Prefer an explicit
featurizer recipe or zoo blocks (see :doc:`model_inputs`):

.. code-block:: text

   normalizedProteomics:fingerprints:randomForest

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Legacy flat key
     - Modern replacement
   * - ``cell_line_views: [gene_expression]``
     - ``scaledGeneExpression`` / ``landmarkGeneExpression``
   * - ``cell_line_views: [proteomics]``
     - ``normalizedProteomics``
   * - ``cell_line_views: [methylation]``
     - ``pca[methylation]`` (+ ``n_components``)
   * - unknown view name
     - ``raw[view]``
   * - ``drug_views: [fingerprints]``
     - ``fingerprints``

``hpam_tune`` naming
~~~~~~~~~~~~~~~~~~~

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

multiprocessing
~~~~~~~~~~~~~~~

Before 1.6.0, ``multiprocessing=True`` selected a parallel HPO path. It now
only emits a warning and does **not** control Ray/Optuna tuning. This remains
available for backward compatibility, but is deprecated and may be removed in
a future release. Prefer ``hyperparameter_tuning=True`` and
``hpo_num_samples``.
