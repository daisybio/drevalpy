Hyperparameter migration
========================

DrEvalPy no longer uses YAML grid files such as
``components/predictors/baselines/hyperparameters.yaml`` for baseline tuning.

Instead, each component predictor owns:

- ``get_default_hyperparameters()`` for the public flat dict passed to ``build_model()``
- ``get_hyperparameter_space()`` for structured Ray + Optuna search

Public API
----------

- ``DRPModel.get_hyperparameter_set()`` returns a single default configuration.
- ``DRPModel.get_structured_hyperparameter_space()`` exposes the tunable search space.
- Experiment tuning uses ``hyperparameter_tuning=True`` with ``hpo_num_samples``,
  ``hpo_random_state``, and ``hpo_resources_per_trial``.

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

Defaults for ``build_model()`` still come from the predictor's
``get_default_hyperparameters()`` implementation.
