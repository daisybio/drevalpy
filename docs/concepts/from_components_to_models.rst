From components to models
=========================

The :doc:`component_catalog` introduced the available building blocks. This
page supplies the grammar for combining them into a runnable model.

A model has three ordered slots:

.. code-block:: text

   cell-line featurizer : drug featurizer : predictor

Feature-free predictors omit both featurizer slots. All feature-dependent
predictors, including literature ports, declare their required featurizers.

Colons separate the slots. Starting with the three ingredients from the end of
the catalog gives the simplest complete recipe:

.. code-block:: text

   scaledGeneExpression:fingerprints:elasticNet

Read it from left to right: scale gene expression for each cell line, compute
drug fingerprints, then fit an elastic-net predictor. The architecture is
fully determined by those three names.

Other single-view recipes follow the same pattern:

.. code-block:: text

   normalizedProteomics:fingerprints:randomForest
   landmarkGenes:fingerprints:xgboost
   scaledGeneExpression:identity:singleDrugElasticNet

For a feature-based single-drug predictor, ``identity`` has routing semantics:
it creates/selects one estimator per drug. The one-hot identity vector is not
concatenated with the cell-line features seen by that estimator. Single-drug
literature predictors such as ``molir`` and ``superfeltr`` use their configured
``identity`` drug featurizer for routing.

Composition is validated before training. Each featurizer declares a
``FeatureFormat`` (numeric matrix, graph, or ragged sequence); each predictor
declares which formats and which input interface
(``FeatureFreePredictor``, ``MatrixPredictor``, or ``BlockPredictor``) it
accepts. Matrix predictors reject graph/ragged payloads, while block
predictors consume the corresponding fitted blocks. An incompatible recipe
fails early rather than reaching the training loop.

Predictor-only recipes
----------------------

Feature-free predictors omit both featurizer slots:

.. code-block:: text

   naiveMean

Its zoo YAML has only ``predictor`` (and optional ``scope`` / hyperparameters).
A bare feature-dependent predictor name without featurizers is rejected.

Qualifying an omics view
------------------------

The ``raw`` and ``pca`` cell-line featurizers need to know which omics layer to
read. Put that view in brackets as part of the featurizer's qualified name:

.. code-block:: text

   raw[expression]:fingerprints:randomForest
   pca[methylation]:fingerprints:randomForest
   raw[proteomics]:fingerprints:randomForest

Common view aliases include ``expression``, ``methylation``, ``mutations``,
``proteomics``, and ``cnv``. The bracket is meaningful throughout the
configuration: ``pca[expression]`` and ``pca[proteomics]`` are different
qualified featurizers.

Combining multiple representations
----------------------------------

Within a featurizer slot, ``+`` concatenates several featurizers into
``concatFeaturizers``. That is how multi-view models are expressed — not with
special multi-view predictor classes:

.. code-block:: text

   raw[expression]+pca[methylation]:fingerprints:xgboost
   landmarkGenes+normalizedProteomics:fingerprints:lightgbm

The left slot can concatenate several cell-line featurizers; the middle slot
can concatenate drug featurizers the same way when needed. The right slot is
always a single predictor name.

Each qualified featurizer may occur only once per slot. Reusing a base name for
different views is valid, but repeating the same qualified name is not:

.. code-block:: text

   raw[expression]+raw[mutations]       # valid
   pca[expression]+pca[expression]      # invalid

The same architecture can be written as structured YAML. For the first
multi-view recipe above:

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

Recipe, YAML, and named preset
------------------------------

Recipe strings and YAML are two representations of the same component tree.
Recipes are concise and useful for custom stacks; YAML can also carry
descriptions and explicit component settings. A model-zoo preset gives a
validated YAML definition a stable, user-facing name.

CLI and Python workflows normally select that preset name. The
:doc:`model_zoo` is the next step in this sequence: it shows the ready-made
architectures shipped with DrEvalPy and the recipe behind each one.

Architecture and hyperparameters
--------------------------------

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
Structured keys are **dotted** and mirror the composed stack. Featurizer keys
use the same qualified selector as the recipe (including the view bracket when
present):

.. code-block:: text

   predictor.<registryName>.<param>
   featurizer.<registry>.<qualifiedFeaturizer>.<param>

Examples:

.. code-block:: text

   predictor.elasticNet.alpha
   predictor.elasticNet.l1_ratio
   featurizer.cell_line.pca[expression].n_components
   featurizer.cell_line.landmarkGenes.standardize

Flat keys such as ``alpha`` remain valid for constructor defaults; legacy
featurizer aliases (for example ``methylation_n_components``) still work but
are deprecated in favor of the dotted form.

Continue the story
------------------

- **Next:** :doc:`model_zoo` — choose a named, ready-to-run architecture
- **Previous:** :doc:`component_catalog` — look up a registered component name
- :doc:`/python/architecture` — composition details and contracts
- :doc:`/python/model_inputs` — custom views and recipe examples in Python
- :doc:`/python/custom_models` — registering your own components

The compatibility section below is a reference for readers migrating older
configurations; it is not required before continuing to the model zoo.

Backward compatibility
----------------------

Before 1.6.0, many models were effectively monolithic classes with optional
``cell_line_views`` / ``drug_views`` treated as hyperparameters, and baseline
tuning used YAML Cartesian grids. That composition model is gone:

- Prefer recipe strings or zoo featurizer / predictor blocks for architecture.
- Prefer component ``get_hyperparameter_space()`` / dotted keys for search.
- View-as-hyperparameter flat keys remain available but are deprecated; see
  :doc:`/python/hyperparameter_tuning` and :doc:`/python/model_inputs`.
