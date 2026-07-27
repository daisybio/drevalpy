From components to models
=========================

DrEvalPy models are **composed**, not hard-coded end-to-end classes. A runnable
model is a stack of registered components:

.. code-block:: text

   cell-line featurizer(s)  +  drug featurizer(s)  +  predictor

The same stack can be described as a short **recipe string**, as YAML in the
:doc:`model_zoo`, or (in Python) as a ``ModelConfig``. CLI and Python both
select models by zoo preset name; custom stacks use the same recipe grammar.

Featurizers and predictors
--------------------------

- **Cell-line featurizers** turn omics (or other sample features) into the
  tensors or matrices a predictor expects — for example scaled gene
  expression, PCA on methylation, or landmark genes.
- **Drug featurizers** do the same for compounds — fingerprints, SMILES-based
  embeddings, molecular graphs, and so on.
- **Predictors** map the featurized inputs to a response (Elastic Net, random
  forest, neural nets, naive baselines, …).

Components are registered by name. Compatibility is checked at composition
time: each featurizer declares a feature kind (``dense``, ``graph``, or
``sequence``), and each predictor declares what kinds it accepts.

The :doc:`model_zoo` lists named presets built from these components. The
full registry of built-in names is in :doc:`/python/component_catalog`.

Recipe strings
--------------

A recipe has three colon-separated slots:

.. code-block:: text

   cellLineFeaturizer:drugFeaturizer:predictor

Examples:

.. code-block:: text

   scaledGeneExpression:fingerprints:elasticNet
   normalizedProteomics:fingerprints:randomForest
   landmarkGeneExpression:fingerprints:xgboost

Some cell-line featurizers take an omics **view** in brackets:

.. code-block:: text

   raw[expression]:fingerprints:randomForest
   pca[methylation]:fingerprints:randomForest
   raw[mynewdatamodality]:fingerprints:randomForest

Common view aliases include ``expression``, ``methylation``, ``mutations``,
``proteomics``, and ``cnv``. Zoo YAML uses the same atoms as the recipe
string (``name`` / ``view`` blocks instead of a single line).

Concatenation with ``+``
~~~~~~~~~~~~~~~~~~~~~~~~

Within a featurizer slot, ``+`` concatenates several featurizers into
``concatFeaturizers``. That is how multi-view models are expressed — not with
special multi-view predictor classes:

.. code-block:: text

   raw[expression]+pca[methylation]:fingerprints:xgboost
   landmarkGeneExpression+normalizedProteomics:fingerprints:lightgbm

The left slot can concatenate several cell-line featurizers; the middle slot
can concatenate drug featurizers the same way when needed. The right slot is
always a single predictor name.

Equivalent zoo YAML for a concatenated cell-line stack:

.. code-block:: yaml

   cell_line_featurizer:
     name: concatFeaturizers
     featurizers:
       - name: raw
         view: expression
       - name: pca
         view: methylation
   drug_featurizer: fingerprints
   predictor: xgboost

Architecture vs hyperparameters
-------------------------------

The recipe (which featurizers and which predictor) is part of the **model
architecture**. It is fixed when the model is composed. Hyperparameter
optimization does **not** search over alternative recipes or omics views; it
searches numeric/categorical parameters *on top of* a fixed stack.

Hyperparameter spaces
---------------------

Each tunable component owns:

- default hyperparameters used when tuning is off
- a structured search space used when tuning is on

Search uses Ray Tune with Optuna as the sampler (see
:doc:`Hyperparameter tuning (CLI) </cli/hyperparameter_tuning>` and
:doc:`Hyperparameter tuning (Python API) </python/hyperparameter_tuning>`).
Structured keys are **dotted** and mirror the composed stack:

.. code-block:: text

   predictor.<registryName>.<param>
   featurizer.<registry>.<featurizerName>.<index>.<param>

Examples for ``scaledGeneExpression:fingerprints:elasticNet``:

.. code-block:: text

   predictor.elasticNet.alpha
   predictor.elasticNet.l1_ratio

The integer after a featurizer name is a **zero-based occurrence index** of
that name in the slot (required in structured keys). A single ``pca`` is
always ``…pca.0.…``. With concatenation and two ``landmarkGenes`` children you
get ``0`` and ``1``:

.. code-block:: text

   featurizer.cell_line.pca.0.n_components
   featurizer.cell_line.landmarkGenes.0.standardize
   featurizer.cell_line.landmarkGenes.1.minmax_scale

Flat keys such as ``alpha`` remain valid for ``build_model``-style defaults;
legacy featurizer aliases (for example ``methylation_n_components``) still
work but are deprecated in favor of the dotted form.

Further reading
---------------

- :doc:`model_zoo` — named presets and their descriptions
- :doc:`/python/architecture` — composition details and contracts
- :doc:`/python/model_inputs` — custom views and recipe examples in Python
- :doc:`/python/component_catalog` — registered featurizer and predictor names
- :doc:`/python/custom_models` — registering your own components

Backward compatibility
----------------------

Before 1.6.0, many models were effectively monolithic classes with optional
``cell_line_views`` / ``drug_views`` treated as hyperparameters, and baseline
tuning used YAML Cartesian grids. That composition model is gone:

- Prefer recipe strings or zoo featurizer / predictor blocks for architecture.
- Prefer component ``get_hyperparameter_space()`` / dotted keys for search.
- View-as-hyperparameter flat keys remain available but are deprecated; see
  :doc:`/python/hyperparameter_tuning` and :doc:`/python/model_inputs`.
