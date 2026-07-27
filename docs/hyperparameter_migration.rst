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
  For featurizers, the integer after the name is a **zero-based occurrence
  index** of that featurizer name in the composed stack (per registry). A single
  ``pca`` is always ``…pca.0.…``; with ``concatFeaturizers`` and several of the
  same name you get ``0``, ``1``, …. The index is **required** in structured
  dotted keys — keys without it are not applied. Flat ``build_model`` dicts
  still use names without an index (e.g. ``n_components``). See
  :doc:`model_architecture` for more examples.
- Experiment tuning uses ``hyperparameter_tuning=True`` with ``hpo_num_samples``,
  ``hpo_random_state``, and ``hpo_resources_per_trial`` (Ray Tune + Optuna).

When ``hyperparameter_tuning=False``, experiments use defaults only. This is **not**
a debug mode and does not iterate legacy grid entries.

Configuring inputs
------------------

Use one of:

- Zoo presets in ``drevalpy/models/zoo/*.yaml`` with explicit
  ``cell_line_featurizer`` / ``drug_featurizer`` blocks
- ``ModelConfig.from_spec()`` / ``ModelConfig.from_yaml()``
- Recipe strings such as ``raw[proteomics]:fingerprints:randomForest``

In a recipe, ``:`` separates cell-line featurizer, drug featurizer, and
predictor. Within a featurizer slot, ``+`` concatenates several featurizers into
``concatFeaturizers`` (for example
``raw[expression]+pca[methylation]:fingerprints:xgboost``). Multi-view models
such as ``MultiViewXGBoost`` and ``MultiViewLightGBM`` are expressed this way
(or with equivalent zoo YAML blocks), not by special multi-view predictor
classes. See :doc:`model_architecture` for the full recipe grammar.

Migrating flat ``build_model`` view keys
----------------------------------------

Through version 1.5.1, ``cell_line_views`` / ``drug_views`` were treated as
hyperparameters. Inputs are now part of the model architecture (recipe / zoo
featurizer blocks), not something HPO retunes. The old keys still work but
emit a ``FutureWarning``:

.. code-block:: yaml

    # Old hpam / build_model YAML (still works, warns)
    cell_line_views: [proteomics]
    drug_views: [fingerprints]

Replace with an explicit featurizer recipe or zoo blocks:

.. code-block:: text

    normalizedProteomics:fingerprints:randomForest

.. code-block:: yaml

    # Zoo YAML
    cell_line_featurizer: normalizedProteomics
    drug_featurizer: fingerprints
    predictor: randomForest

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

See :doc:`example_flexible_inputs` and :doc:`model_architecture`.

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
