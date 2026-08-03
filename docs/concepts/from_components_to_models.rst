From components to models
=========================

The :doc:`component_catalog` introduced the available building blocks. This
page supplies the grammar for combining them into a runnable model.

A model consists of three components:

- a cell-line featurizer
- a drug featurizer
- a predictor

DrEvalPy provides multiple ways of defining which components should be used in
a model:

- **Recipe strings** — concise; do not carry hyperparameter spaces
- **YAML files** — more verbose; can declare hyperparameter spaces
- **ModelConfig** — same information as YAML, but Python-native

Examples below use a tab switcher so you can compare all three notations for
the same architecture.

Basic composition
-----------------

The three slots are always cell-line featurizer, drug featurizer, then
predictor:

.. tab-set::
   :sync-group: composition

   .. tab-item:: Recipe string
      :sync: recipe

      .. code-block:: text

         cell-line featurizer : drug featurizer : predictor

   .. tab-item:: YAML
      :sync: yaml

      .. code-block:: yaml

         cell_line_featurizer: <name>
         drug_featurizer: <name>
         predictor: <name>

   .. tab-item:: ModelConfig
      :sync: modelconfig

      .. code-block:: python

         from drevalpy.models.config import (
             CellLineFeaturizerConfig,
             DrugFeaturizerConfig,
             ModelConfig,
             PredictorConfig,
         )

         config = ModelConfig(
             cell_line_featurizer=CellLineFeaturizerConfig(name="<name>"),
             drug_featurizer=DrugFeaturizerConfig(name="<name>"),
             predictor=PredictorConfig(name="<name>"),
         )

A very simple complete stack is gene-expression scaling, drug fingerprints,
and an elastic-net predictor:

.. tab-set::
   :sync-group: composition

   .. tab-item:: Recipe string
      :sync: recipe

      .. code-block:: text

         scaledGeneExpression:fingerprints:elasticNet

   .. tab-item:: YAML
      :sync: yaml

      .. code-block:: yaml

         cell_line_featurizer: scaledGeneExpression
         drug_featurizer: fingerprints
         predictor: elasticNet

   .. tab-item:: ModelConfig
      :sync: modelconfig

      .. code-block:: python

         from drevalpy.models.config import (
             CellLineFeaturizerConfig,
             DrugFeaturizerConfig,
             ModelConfig,
             PredictorConfig,
         )

         config = ModelConfig(
             cell_line_featurizer=CellLineFeaturizerConfig(
                 name="scaledGeneExpression"
             ),
             drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
             predictor=PredictorConfig(name="elasticNet"),
         )

Read it from left to right: scale gene expression for each cell line, compute
drug fingerprints, then fit an elastic-net predictor. The architecture is
fully determined by those three names.

Other single-view stacks follow the same pattern:

.. tab-set::
   :sync-group: composition

   .. tab-item:: Recipe string
      :sync: recipe

      .. code-block:: text

         normalizedProteomics:fingerprints:randomForest
         landmarkGenes:fingerprints:xgboost
         scaledGeneExpression:identity:singleDrugElasticNet

   .. tab-item:: YAML
      :sync: yaml

      .. code-block:: yaml

         cell_line_featurizer: normalizedProteomics
         drug_featurizer: fingerprints
         predictor: randomForest

      .. code-block:: yaml

         cell_line_featurizer: landmarkGenes
         drug_featurizer: fingerprints
         predictor: xgboost

      .. code-block:: yaml

         cell_line_featurizer: scaledGeneExpression
         drug_featurizer: identity
         predictor: singleDrugElasticNet

   .. tab-item:: ModelConfig
      :sync: modelconfig

      .. code-block:: python

         from drevalpy.models.config import (
             CellLineFeaturizerConfig,
             DrugFeaturizerConfig,
             ModelConfig,
             PredictorConfig,
         )

      .. code-block:: python

         config = ModelConfig(
             cell_line_featurizer=CellLineFeaturizerConfig(
                 name="normalizedProteomics"
             ),
             drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
             predictor=PredictorConfig(name="randomForest"),
         )

      .. code-block:: python

         config = ModelConfig(
             cell_line_featurizer=CellLineFeaturizerConfig(
                 name="landmarkGenes"
             ),
             drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
             predictor=PredictorConfig(name="xgboost"),
         )

      .. code-block:: python

         config = ModelConfig(
             cell_line_featurizer=CellLineFeaturizerConfig(
                 name="scaledGeneExpression"
             ),
             drug_featurizer=DrugFeaturizerConfig(name="identity"),
             predictor=PredictorConfig(name="singleDrugElasticNet"),
         )

In the last example, ``singleDrugElasticNet`` uses the ``identity`` drug
featurizer, which one-hot encodes drug identifiers, to create a single
estimator per drug.

Featurizers that can operate on multiple omics layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``raw`` and ``pca`` cell-line featurizers are flexible towards which omics
layer to read. Put that view in brackets as part of the featurizer's
qualified name:

.. tab-set::
   :sync-group: composition

   .. tab-item:: Recipe string
      :sync: recipe

      .. code-block:: text

         raw[expression]:fingerprints:randomForest
         pca[methylation]:fingerprints:randomForest
         raw[proteomics]:fingerprints:randomForest

   .. tab-item:: YAML
      :sync: yaml

      .. code-block:: yaml

         cell_line_featurizer:
           name: raw
           view: expression
         drug_featurizer: fingerprints
         predictor: randomForest

         # pca[methylation]:fingerprints:randomForest
         # cell_line_featurizer:
         #   name: pca
         #   view: methylation
         # drug_featurizer: fingerprints
         # predictor: randomForest

   .. tab-item:: ModelConfig
      :sync: modelconfig

      .. code-block:: python

         from drevalpy.models.config import (
             CellLineFeaturizerConfig,
             DrugFeaturizerConfig,
             ModelConfig,
             PredictorConfig,
         )

         config = ModelConfig(
             cell_line_featurizer=CellLineFeaturizerConfig(
                 name="raw",
                 view="expression",
             ),
             drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
             predictor=PredictorConfig(name="randomForest"),
         )

         # pca[methylation]:fingerprints:randomForest
         # ModelConfig(
         #     cell_line_featurizer=CellLineFeaturizerConfig(
         #         name="pca", view="methylation"
         #     ),
         #     drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
         #     predictor=PredictorConfig(name="randomForest"),
         # )

Common view aliases include ``expression``, ``methylation``, ``mutations``,
``proteomics``, and ``cnv``.

Combining multiple representations
----------------------------------

Within a featurizer slot, ``+`` concatenates several featurizers into
``concatFeaturizers``:

.. tab-set::
   :sync-group: composition

   .. tab-item:: Recipe string
      :sync: recipe

      .. code-block:: text

         raw[expression]+pca[methylation]:fingerprints:xgboost
         landmarkGenes+normalizedProteomics:fingerprints:lightgbm

   .. tab-item:: YAML
      :sync: yaml

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

   .. tab-item:: ModelConfig
      :sync: modelconfig

      .. code-block:: python

         from drevalpy.models.config import (
             CellLineFeaturizerConfig,
             DrugFeaturizerConfig,
             ModelConfig,
             PredictorConfig,
         )

         config = ModelConfig(
             cell_line_featurizer=CellLineFeaturizerConfig(
                 name="concatFeaturizers",
                 hyperparameters={
                     "featurizers": [
                         {"name": "raw", "view": "expression"},
                         {"name": "pca", "view": "methylation"},
                     ],
                 },
             ),
             drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
             predictor=PredictorConfig(name="xgboost"),
         )

The left slot can concatenate several cell-line featurizers; the middle slot
can concatenate drug featurizers the same way when needed. The right slot is
always a single predictor name.

Composition validation
----------------------

Composition is validated before training. Each featurizer declares a
``FeatureFormat`` (numeric matrix, graph, or ragged sequence); each predictor
declares which formats and which input interface
(``FeatureFreePredictor``, ``MatrixPredictor``, or ``BlockPredictor``) it
accepts. Matrix predictors reject graph/ragged payloads, while block
predictors consume the corresponding fitted blocks. An incompatible recipe
fails early rather than reaching the training loop.

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
