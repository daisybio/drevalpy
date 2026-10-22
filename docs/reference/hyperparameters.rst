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
- An alias. ``methylation_n_components`` and ``methylation_pca_components`` both name
  ``sample_featurizer.pca[methylation].n_components``.

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
