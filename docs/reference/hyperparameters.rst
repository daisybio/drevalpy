Hyperparameter keys and search spaces
=====================================

Every component of a model declares hyperparameters, each with a search space. This page
describes the spaces, the keys that name a hyperparameter in a composed model, and how a
YAML file overrides them.

Search-space entries
--------------------

A search space maps a parameter name to an entry. Every entry is a mapping with a ``default``.
A space that contains anything else raises ``ValueError``.

.. list-table::
   :header-rows: 1
   :widths: 16 34 50

   * - ``type``
     - Fields
     - Value drawn
   * - ``int``
     - ``low``, ``high``, optional ``log``
     - An integer from ``low`` to ``high``
   * - ``float``
     - ``low``, ``high``, optional ``log``
     - A float from ``low`` to ``high``
   * - ``categorical``
     - ``choices``
     - One element of ``choices``
   * - ``pow2``
     - ``low``, ``high``
     - ``2 ** e`` for an integer exponent ``e`` from ``low`` to ``high``

A choice may be a list, such as the layer widths ``[32, 16, 8, 4]`` of the ``neuralNetwork``
predictor; Optuna draws it by position and the model receives the list itself.

``low`` and ``high`` are inclusive. ``log: true`` samples on a logarithmic scale. For
``pow2`` the bounds are exponents: ``{type: pow2, low: 9, high: 11}`` draws 512, 1024 or 2048,
and its ``default`` is written as the value, such as ``2048``. An entry without a ``type``
is treated as ``categorical``. Any other ``type`` raises ``ValueError`` when a value is drawn.

Example:

.. code-block:: yaml

   alpha:
     type: float
     low: 0.0001
     high: 10.0
     log: true
     default: 1.0

Qualified keys
--------------

In a composed model, each hyperparameter has a qualified key that names the component and the
parameter.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Key
     - Component
   * - ``predictor.<name>.<param>``
     - The predictor
   * - ``sample_featurizer.<selector>.<param>``
     - A sample featurizer
   * - ``drug_featurizer.<selector>.<param>``
     - A drug featurizer

``<selector>`` is the featurizer name, or ``name[view]`` for a featurizer with a view. A
``concatFeaturizers`` node has no keys of its own: each child is keyed by its own selector.

For the recipe ``pca[gene_expression]:fingerprints:elasticNet`` the space has these keys:

.. code-block:: text

   sample_featurizer.pca[gene_expression].n_components
   drug_featurizer.fingerprints.radius
   drug_featurizer.fingerprints.n_bits
   drug_featurizer.fingerprints.use_chirality
   drug_featurizer.fingerprints.use_counts
   predictor.elasticNet.alpha
   predictor.elasticNet.l1_ratio

``ModelSpec.hyperparameter_space`` returns this mapping, and ``ModelSpec.default_hyperparameters``
returns the defaults under the short parameter names.

Keys that DrEvalPy accepts
--------------------------

``ModelSpec.resolve(hyperparameters)`` and ``DRPModel`` take a flat mapping. Each key is one of:

- A qualified key.
- A short parameter name such as ``alpha``, if exactly one component of the model has it.
- An alias. ``methylation_n_components`` (the 1.x name) names
  ``sample_featurizer.pca[methylation].n_components``. It is an error for a model without a
  ``pca[methylation]`` featurizer, such as the ``MultiView*`` presets, which read gene expression
  and mutations only.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Input
     - Result
   * - A short name shared by several components
     - ``ValueError``: ``Ambiguous hyperparameter … Use one of: …``
   * - A name no component has
     - ``ValueError``: ``Unknown hyperparameter … for this model stack.``
   * - Two keys for the same parameter
     - ``ValueError``: ``Duplicate hyperparameter assignment …``
   * - The old indexed form ``sample_featurizer.<name>.<index>.<param>``
     - ``ValueError``: ``Indexed featurizer hyperparameter keys are no longer supported``

Defaults of the shipped presets
-------------------------------

Every preset starts from the first value of each grid in the ``hyperparameters.yaml`` files of
DrEvalPy 1.6.1. Where a parameter had no entry there, the 1.6.1 code default applies. The search
spaces are wider than 1.6.1 grids, so each grid value stays inside its range.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Preset
     - Defaults
   * - ``ElasticNet``
     - ``alpha=1``, ``l1_ratio=0`` (which fits a ridge regression); other solver settings are those
       of scikit-learn
   * - ``SingleDrugElasticNet``
     - ``alpha=1``, ``l1_ratio=0.2``
   * - ``Lasso``
     - ``alpha=1``, ``max_iter=10000``, ``tol=0.001``, ``selection=random`` with ``random_state=0``
       (1.6.1 left the seed unset)
   * - ``Ridge``
     - ``alpha=1``
   * - ``RandomForest``, ``SingleDrugRandomForest``, ``MultiViewRandomForest``
     - ``n_estimators=100``, ``max_depth=5``, ``max_samples=0.2``
   * - ``GradientBoosting``
     - ``max_depth=5``, ``learning_rate=0.1``, ``max_iter=100``, ``random_state=42``,
       ``validation_fraction=0.1``, ``n_iter_no_change=10``
   * - ``AdaBoostDecisionTree``
     - ``max_depth=4``, ``n_estimators=50``, ``min_samples_split=2``, ``min_samples_leaf=1``
   * - ``SVR``
     - ``C=0.001``, ``epsilon=0.001``, ``max_iter=500``
   * - ``KNNRegressor``
     - ``n_neighbors=5``, ``weights=distance``
   * - ``MultiViewXGBoost``
     - ``max_depth=10``, ``subsample=0.8``, ``colsample_bytree=0.6``, ``reg_alpha=0``, ``reg_lambda=0.1``
   * - ``MultiViewLightGBM``
     - ``num_leaves=31``, ``subsample=0.8``, ``colsample_bytree=0.6``, ``reg_alpha=0``, ``reg_lambda=0``
   * - ``SimpleNeuralNetwork``
     - ``units_per_layer=[32, 16, 8, 4]``, ``dropout_prob=0.3``, ``max_epochs=100``
   * - ``MultiViewNeuralNetwork``
     - as ``SimpleNeuralNetwork`` with ``units_per_layer=[16, 8, 4]``
   * - ``DrugGNN``
     - ``epochs=100``, ``batch_size=1024``, ``hidden_dim=64``, ``dropout=0.2``
   * - ``PharmaFormer``
     - ``gene_hidden_size=8192``, ``drug_hidden_size=256``, ``feature_dim=128``, ``nhead=8``,
       ``num_layers=3``, ``dim_feedforward=2048``, ``dropout=0.1``, ``batch_size=128``
   * - ``SparseGO``
     - ``num_neurons_drug=[100, 50, 6]``, ``num_neurons_final=12``, ``num_neurons_per_final_GO=30``

The presets also read the gene lists of 1.6.1, so a dataset must hold every gene of the list:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Preset
     - Gene list
   * - All sklearn presets, ``SimpleNeuralNetwork``, ``PharmaFormer``, ``DIPK``, ``DrugGNN``
     - ``landmark_genes`` of gene expression
   * - ``MultiView*``
     - ``landmark_genes`` of gene expression and ``drug_target_genes_all_drugs`` of mutations
   * - ``MOLIR``
     - ``gene_expression_intersection``, ``mutations_intersection`` and
       ``copy_number_variation_gistic_intersection``
   * - ``SuperFELTR``
     - every gene

Overriding a space
------------------

A component given as a mapping can carry a ``hyperparameter_space``. It replaces the
component's built-in space entirely: a parameter left out of it is no longer tuned and is not
part of the model's keys.

.. code-block:: yaml

   predictor:
     name: elasticNet
     hyperparameter_space:
       alpha: {type: float, low: 0.01, high: 1.0, log: true, default: 0.1}
       l1_ratio: {type: float, low: 0.0, high: 1.0, default: 0.5}

The same field exists on featurizers; see :doc:`recipes` for the featurizer forms. A loose
value in the one-key form, such as ``{elasticNet: {alpha: 0.5}}``, sets the ``default`` of a
declared parameter and does not change its range.

Tuning
------

With tuning enabled, ``drevalpy run``, ``drevalpy single``, ``drevalpy.run`` and
``drevalpy.single`` draw each trial from the model's space with an Optuna ``TPESampler``.

.. list-table::
   :header-rows: 1
   :widths: 30 24 46

   * - Option
     - Default
     - Meaning
   * - ``hpo_metric`` / ``--hpo-metric``
     - ``RMSE``
     - Metric to optimize; minimized for MSE, RMSE and MAE, maximized for the others
   * - ``hpo_num_trials`` / ``--hpo-trials``
     - ``16``
     - Number of trials
   * - ``hpo_random_state`` / ``--hpo-random-state``
     - the run seed
     - Seed of the sampler
   * - ``hyperparameter_tuning`` / ``--hpo``
     - on
     - ``--no-hpo`` fits with the defaults

The values found are stored in each run's ``best_hyperparameters``; see :doc:`results`.
