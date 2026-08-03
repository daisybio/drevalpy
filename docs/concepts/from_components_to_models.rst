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
-----------------------------------------------------

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

      .. code-block:: yaml

         cell_line_featurizer:
           name: pca
           view: methylation
         drug_featurizer: fingerprints
         predictor: randomForest

      .. code-block:: yaml

         cell_line_featurizer:
           name: raw
           view: proteomics
         drug_featurizer: fingerprints
         predictor: randomForest

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
                 name="raw",
                 view="expression",
             ),
             drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
             predictor=PredictorConfig(name="randomForest"),
         )

      .. code-block:: python

         config = ModelConfig(
             cell_line_featurizer=CellLineFeaturizerConfig(
                 name="pca",
                 view="methylation",
             ),
             drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
             predictor=PredictorConfig(name="randomForest"),
         )

      .. code-block:: python

         config = ModelConfig(
             cell_line_featurizer=CellLineFeaturizerConfig(
                 name="raw",
                 view="proteomics",
             ),
             drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
             predictor=PredictorConfig(name="randomForest"),
         )

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

      .. code-block:: yaml

         cell_line_featurizer:
           - landmarkGenes
           - normalizedProteomics
         drug_featurizer: fingerprints
         predictor: lightgbm

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

      .. code-block:: python

         config = ModelConfig(
             cell_line_featurizer=CellLineFeaturizerConfig(
                 name="concatFeaturizers",
                 hyperparameters={
                     "featurizers": [
                         "landmarkGenes",
                         "normalizedProteomics",
                     ],
                 },
             ),
             drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
             predictor=PredictorConfig(name="lightgbm"),
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

Hyperparameter spaces
---------------------

Only the YAML and ModelConfig interfaces allow specifying hyperparameter
spaces. Recipe strings describe architecture only; when you use a recipe,
each component falls back to its built-in hyperparameter space.

On a YAML or ModelConfig stack, set ``hyperparameter_space`` on a component to
**replace** that component's built-in search space. Specs use local parameter
names (``alpha``, ``n_components``, …); DrEvalPy prefixes them for tuning.

.. tab-set::
   :sync-group: composition

   .. tab-item:: YAML
      :sync: yaml

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
             drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
             predictor=PredictorConfig(
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

Hyperparameter names during search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When tuning runs, Ray Tune / Optuna see a **merged** space whose keys are
dotted and mirror the composed stack:

.. code-block:: text

   predictor.<registryName>.<param>
   cell_line_featurizer.<qualifiedFeaturizer>.<param>
   drug_featurizer.<qualifiedFeaturizer>.<param>

The featurizer selector is the same qualified name as in a recipe, including
the view bracket when present (``pca[expression]``, ``landmarkGenes``, …).
For the example above, that yields:

.. code-block:: text

   predictor.elasticNet.alpha
   predictor.elasticNet.l1_ratio
   cell_line_featurizer.pca[expression].n_components

The same base featurizer on different views stays independently tunable
(``pca[expression]`` vs ``pca[proteomics]``). Repeating the same qualified
selector in one slot is rejected.

Continue the story
------------------

- **Next:** :doc:`model_zoo` — choose a named, ready-to-run architecture
- **Previous:** :doc:`component_catalog` — look up a registered component name
- :doc:`/python/architecture` — composition details and contracts
- :doc:`/python/hyperparameter_tuning` — running search on a fixed stack

