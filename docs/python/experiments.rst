Experiments and Hyperparameter Optimization
===========================================

If you are reading this, we assume you are already familiar with these
concepts:

- :doc:`/concepts/evaluation`
- :doc:`/concepts/from_components_to_models`

The experiment pipeline
-----------------------

DrEvalPy provides two levels of experiment execution:

- :func:`~drevalpy.run` — orchestrates models × folds × randomization in
  one call. Returns an :class:`~drevalpy.types.results.ExperimentResult`.
- :func:`~drevalpy.single` — trains a single model on a single fold.
  Returns a :class:`~drevalpy.types.results.RunResult`. Used when you need
  per-fold control (or parallelism via the CLI).

For day-to-day benchmarking, prefer ``run`` over a hand-rolled train/predict
loop on individual models (see :doc:`models` for the low-level lifecycle).

Minimal call
------------

.. code-block:: python

   from drevalpy.data import load
   from drevalpy.models import construct_model
   from drevalpy import run

   dataset = load("GDSC1")
   ElasticNet = construct_model("ElasticNet")

   result = run(
       models=[ElasticNet],
       dataset=dataset,
       split_mode="LCO",
       hyperparameter_tuning=False,
   )

Pass model **classes** from :func:`~drevalpy.models.construct_model`, not
instances. The result is an
:class:`~drevalpy.types.results.ExperimentResult` containing all fold
predictions and metrics.

``run`` parameters
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``models``
     -
     - List of ``DRPModel`` subclasses to evaluate.
   * - ``dataset``
     -
     - ``Dataset`` object or name string (auto-loaded).
   * - ``split_mode``
     -
     - Split mode: ``"LPO"``, ``"LCO"``, ``"LDO"``, or ``"LTO"`` (see :doc:`/concepts/evaluation`).
   * - ``hyperparameter_tuning``
     - ``True``
     - Whether to run HPO on each model.
   * - ``hpo_metric``
     - ``"RMSE"``
     - Metric to optimize (also ``MSE``, ``MAE``, ``R^2``, ``Pearson``, ``Spearman``, ``Kendall``).
   * - ``hpo_num_samples``
     - ``16``
     - Number of Optuna trials per fold.
   * - ``hpo_random_state``
     - ``42``
     - Seed for the Optuna sampler.
   * - ``randomization_modes``
     - ``None``
     - Optional list of feature-shuffle modes: ``"SVCC"``, ``"SVRC"``, ``"SVCD"``, ``"SVRD"``.
   * - ``randomization_type``
     - ``"permutation"``
     - ``"permutation"`` or ``"invariant"``.
   * - ``robustness_trials``
     - ``0``
     - Number of shuffled robustness repetitions (0 = disabled).
   * - ``precomputed_only``
     - ``False``
     - Restrict HPO to pre-computed featurizer variants.

Per-fold execution: ``single``
------------------------------

For parallel or custom workflows, split first and execute folds individually:

.. code-block:: python

   from drevalpy.data import load, split
   from drevalpy.models import construct_model
   from drevalpy import single

   dataset = load("GDSC1")
   folds = split(dataset, mode="LCO", n_splits=5)
   ElasticNet = construct_model("ElasticNet")

   results = []
   for fold in folds:
       result = single(
           ElasticNet,
           dataset,
           fold,
           hyperparameter_tuning=True,
           hpo_metric="RMSE",
           hpo_num_samples=16,
       )
       results.append(result)

Aggregate into an :class:`~drevalpy.types.results.ExperimentResult`:

.. code-block:: python

   from drevalpy.types.results import ExperimentResult

   experiment = ExperimentResult(results)
   experiment.save("results/")

``single`` additionally supports:

- ``response_transformation`` — an sklearn ``TransformerMixin`` (``standard``,
  ``minmax``, ``robust``) applied to target values during training and inverted
  for final predictions.

Hyperparameter tuning
---------------------

Tuning is **on by default**. When ``hyperparameter_tuning=True``, each model
with a search space is tuned with Ray Tune and Optuna before final fold
evaluation. Set ``hyperparameter_tuning=False`` to skip search and use each
model's ``get_default_hyperparameters()`` only.

Ray and Optuna
~~~~~~~~~~~~~~

Ray and Optuna fulfill different roles in the hyperparameter tuning process:

- **Ray Tune** runs and schedules trials (parallelism, resource allocation,
  trial storage under the run directory).
- **Optuna** (via Ray's ``OptunaSearch``) chooses which hyperparameter values
  to try next and optimizes ``hpo_metric``.

.. note::

   Ray is generally installed as a dependency of drevalpy.
   However, Ray is only compatible with a limited set of Python versions on
   Windows. See
   :ref:`getting_started/installation:Hyperparameter tuning on Windows`
   for more details.

Search spaces
~~~~~~~~~~~~~

The hyperparameter search space of a model is fixed when you
:func:`~drevalpy.models.construct_model` the class.
Whether you can customize the space depends on how you construct the class:

- **Recipe strings** always use each component's built-in search space; they
  cannot express overrides.
- **Built-in zoo presets** ship as YAML inside the ``drevalpy`` package, so
  you cannot override their search space from the call site. A preset may
  still deviate from each component's built-in defaults when its YAML sets
  ``hyperparameter_space``; otherwise it falls back to those defaults.
- **YAML** (via ``config.from_yaml``) and the **``ModelConfig``
  constructor** let you set ``hyperparameter_space`` on a component to
  **replace** its built-in space. Use these when you need a custom search
  space (including your own zoo-style YAML files).

.. tab-set::

   .. tab-item:: YAML

      .. code-block:: yaml

         cell_line_featurizer:
           name: pca
           view: expression
           hyperparameter_space:
             n_components:
               type: int
               low: 8
               high: 512
               default: 128
         drug_featurizer: fingerprints
         predictor:
           name: elasticNet
           hyperparameter_space:
             alpha:
               type: float
               low: 1.0e-4
               high: 10.0
               log: true
               default: 1.0
             l1_ratio:
               type: float
               low: 0.0
               high: 1.0
               default: 0.5

      .. code-block:: python

         from drevalpy.models import config, construct_model

         cfg = config.from_yaml("my_zoo/custom_en.yaml")
         MyEN = construct_model("MyElasticNet", cfg)

   .. tab-item:: ModelConfig

      .. code-block:: python

         from drevalpy.models import config, construct_model

         cfg = config.ModelConfig(
             cell_line_featurizer=config.CellLineFeaturizerConfig(
                 name="pca",
                 view="expression",
                 hyperparameter_space={
                     "n_components": {
                         "type": "int",
                         "low": 8,
                         "high": 512,
                         "default": 128,
                     },
                 },
             ),
             drug_featurizer=config.DrugFeaturizerConfig(name="fingerprints"),
             predictor=config.PredictorConfig(
                 name="elasticNet",
                 hyperparameter_space={
                     "alpha": {
                         "type": "float",
                         "low": 1e-4,
                         "high": 10.0,
                         "log": True,
                         "default": 1.0,
                     },
                     "l1_ratio": {
                         "type": "float",
                         "low": 0.0,
                         "high": 1.0,
                         "default": 0.5,
                     },
                 },
             ),
         )
         MyEN = construct_model("MyElasticNet", cfg)

Specs use local parameter names (``alpha``, ``n_components``, …). During
search, Ray Tune / Optuna see dotted qualified keys to prevent name
collisions, for example ``predictor.elasticNet.alpha`` and
``cell_line_featurizer.pca[expression].n_components``. See
:doc:`/concepts/from_components_to_models` for the full key rules and
:doc:`models` for how to construct classes from YAML or ``ModelConfig``.

Randomization and robustness
-----------------------------

Pass ``randomization_modes`` to ``run`` for feature-shuffle tests
(``"SVCC"``, ``"SVRC"``, ``"SVCD"``, ``"SVRD"``; ``None`` disables).
``randomization_type`` is ``"permutation"`` (default) or ``"invariant"``.
``robustness_trials`` repeats training with shuffled fold orderings;
``0`` (default) disables the robustness test.

These extras apply to all ``models`` in the run. See
:doc:`/concepts/evaluation` for the interpretation of randomization and
robustness results.
