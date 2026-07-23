Hyperparameter migration
========================

DrEvalPy no longer uses YAML grid files for baseline tuning.

Instead, each component predictor (and tunable featurizers such as ``pca`` or
``landmarkGenes``) owns:

- ``get_default_hyperparameters()`` for the public flat dict passed to ``build_model()``
- ``get_hyperparameter_space()`` for structured Ray + Optuna search

Public API
----------

- ``DRPModel.get_hyperparameter_set()`` returns a **single** default configuration
  (``[get_default_hyperparameters()]``), not a full Cartesian grid.
- ``DRPModel.get_structured_hyperparameter_space()`` exposes the tunable search space
  with dotted keys (``predictor.elasticNet.alpha``,
  ``featurizer.cell_line.pca.0.n_components``, …).
- Experiment tuning uses ``hyperparameter_tuning=True`` with ``hpo_num_samples``,
  ``hpo_random_state``, and ``hpo_resources_per_trial`` (Ray Tune + Optuna).

When ``hyperparameter_tuning=False``, experiments use defaults only. This is **not**
a debug mode and does not iterate legacy grid entries.

Configuring inputs
------------------

Use one of:

- Zoo presets in ``drevalpy/models/zoo/*.yaml``
- ``ModelConfig.from_spec()`` / ``ModelConfig.from_yaml()``
- Public hyperparameters ``cell_line_views`` and ``drug_views`` for sklearn baselines

Multi-view models such as ``MultiViewXGBoost`` and ``MultiViewLightGBM`` are expressed
by composing featurizers in zoo YAML, not by special multi-view predictor classes.

Translating old YAML grids
--------------------------

Old YAML entries such as:

.. code-block:: yaml

    ElasticNet:
      alpha:
        - 1
        - 0.1
        - 10

become component search spaces, for example:

.. code-block:: python

    {"alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0}}

At experiment time, enable Ray search instead of looping ``get_hyperparameter_set()``:

.. code-block:: python

    drug_response_experiment(..., hyperparameter_tuning=True, hpo_num_samples=16)

Defaults for ``build_model()`` still come from the predictor's
``get_default_hyperparameters()`` implementation.

CLI notes
---------

- ``--no_hyperparameter_tuning`` disables Ray search and uses predictor defaults.
- ``--multiprocessing`` is a deprecated alias; prefer ``--hpo_num_samples`` with tuning enabled.

Dependencies
------------

``ray[tune]``, ``optuna``, and ``pydantic`` are core dependencies (included in the
default ``pip install drevalpy``). See :doc:`installation`.
