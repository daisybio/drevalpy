Experiments and Hyperparameter Optimization
===========================================

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

   from drevalpy.datasets import load_mudataset
   from drevalpy.experiment import mu_experiment
   from drevalpy.models import construct_model

   mudataset = load_mudataset("TOYv1")
   ElasticNet = construct_model("ElasticNet")

   drug_response_experiment(
       models=[ElasticNet],
       response_data=response_data,
       run_id="en_toy",
       test_mode="LCO",
       n_cv_splits=5,
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
- ``baselines``: extra baseline classes; randomization and robustness apply
  only to ``models``.
- ``n_cv_splits``: number of outer CV folds (default ``5``).
- ``path_out`` / ``run_id``: where predictions are written. Built-in
  datasets and features are cached in the system cache directory (see
  :doc:`/getting_started/installation` for ``DREVALPY_CACHE_DIR``).
- ``randomization_mode`` / ``n_trials_robustness``: optional stress tests
  (see :doc:`/concepts/evaluation`).
- ``cross_study_datasets``: other ``MuDataset`` instances for
  transfer evaluation.
- ``wandb_project``: enable Weights & Biases logging when set.

Hyperparameter options are covered below. See :doc:`visualization` for
reports over the written predictions.

Hyperparameter tuning
---------------------

Tuning is **on by default**. When ``hyperparameter_tuning=True``, each model
with a search space is tuned with Ray Tune and Optuna before final fold
evaluation. Set ``hyperparameter_tuning=False`` to skip search and use each
model's ``get_default_hyperparameters()`` only (as in the minimal example
above).

Configure the search with:

- ``hpam_optimization_metric`` — metric Optuna optimizes (default ``RMSE``;
  also ``MSE``, ``MAE``, ``R^2``, ``Pearson``, ``Spearman``, ``Kendall``)
- ``hpo_num_samples`` — number of Optuna trials per fold (default ``16``)
- ``hpo_random_state`` — seed for the Optuna sampler (default ``42``)
- ``hpo_resources_per_trial`` — optional Ray resource dict (for example
  ``{"cpu": 2, "gpu": 0.5}``); GPU defaults apply when CUDA is available

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

Ray and Optuna
~~~~~~~~~~~~~~

Ray and Optuna fulfill different roles in the hyperparameter tuning process:

- **Ray Tune** runs and schedules trials (parallelism,
  ``hpo_resources_per_trial``, trial storage under the run directory).
- **Optuna** (via Ray's ``OptunaSearch``) chooses which hyperparameter values
  to try next and optimizes ``hpam_optimization_metric``.

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
search, Ray Tune / Optuna see dotted qualified keys to prevent name collisions, for example
``predictor.elasticNet.alpha`` and
``cell_line_featurizer.pca[expression].n_components``. See
:doc:`/concepts/from_components_to_models` for the full key rules and
:doc:`models` for how to construct classes from YAML or ``ModelConfig``.
