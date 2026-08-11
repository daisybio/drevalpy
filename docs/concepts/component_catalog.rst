Component catalog
=================

This catalog is the vocabulary of a DrEvalPy model. Every model is assembled
from components with three distinct roles:

- a **cell-line featurizer** represents the biological sample,
- a **drug featurizer** represents the compound, and
- a **predictor** maps those representations to a drug-response estimate.

.. mermaid::

   flowchart LR
      cellLineData["Cell line"]
      drugData["Drug"]
      responseEstimate["Drug response estimate"]

      subgraph model [Model]
         cellLineFeaturizer["Cell-line featurizer"]
         drugFeaturizer["Drug featurizer"]
         predictor["Predictor"]
      end

      cellLineData --> cellLineFeaturizer
      drugData --> drugFeaturizer
      cellLineFeaturizer --> predictor
      drugFeaturizer --> predictor
      predictor --> responseEstimate

The names below are the stable registry names used in recipes and model-zoo
definitions. They are case-sensitive. At this stage, focus on what each
component contributes; the next page, :doc:`from_components_to_models`,
explains how the names fit together and how compatibility is checked.

Featurizers
-----------

Featurizers vs encoders
~~~~~~~~~~~~~~~~~~~~~~~

In DrEvalPy, we distinguish between featurizers and encoders.
Both are used to transform the input data into a feature space that can be used by the predictor, but they differ in their purpose, how they are trained, and how they are used.

**Featurizers** are strategies for extracting features from the input data in an unsupervised manner.
They can be precomputed from cell-line or drug data alone, without the need for drug response labels.
Featurizers can however have hyperparameters, for example the number of principal components to keep in a PCA transformation.
While featurizer representations are generally predictor-agnostic, the optimal hyperparameters can depend on the predictor, the data, and the task.
In DrEvalPy, we try to make sure that featurizers can be elegantly combined with different predictors, while ensuring that hyperparameters are optimized jointly with the predictor.

**Encoders** on the other hand are parts of models that are optimized alongside the main prediction head of a model.
Examples of encoders are the per-omics encoders inside SuperFELTR or the transformer stack inside PharmaFormer.
As these components are tightly coupled to the prediction head and the weights of both are optimized jointly, they are not considered featurizers.
In DrEvalPy, encoders are baked into the predictors and cannot be combined with other predictors.

Precomputable featurizers
~~~~~~~~~~~~~~~~~~~~~~~~~

Some featurizers produce representations that depend only on the entity itself
(the cell line or the drug) and not on any training labels. Their output is
deterministic and identical regardless of which predictor consumes it or which
CV fold is active. These featurizers are marked as **precomputable**.

Precomputable featurizers are typically computationally expensive (model-based
embeddings, graph construction, BPE tokenization). Their results can be stored
inside the dataset ahead of time, so the experiment loop never has to
recompute them. Lightweight featurizers such as ``landmarkGenes`` or
``scaledGeneExpression`` are fast enough that precomputation is unnecessary --
they run inline and are *not* marked as precomputable.

Precomputation and hyperparameter optimization are not mutually exclusive.
A precomputable featurizer can still have hyperparameters (for example, the
fingerprint radius). Because each hyperparameter configuration yields a
different cached output, multiple variants can be stored side by side in the
same dataset. At experiment time, users choose one of two strategies:

- **Precomputed-only mode** -- the optimizer treats the stored variants as
  categorical choices and selects among them. This is fast because no
  featurizer computation happens at all during the search.
- **Standard HPO mode** -- the optimizer ignores stored variants and explores
  the featurizer's continuous hyperparameter space normally, recomputing
  features for each trial. This is slower but can find configurations that
  were never precomputed.

Featurizer types
~~~~~~~~~~~~~~~~

- ``numeric_matrix`` — one dense numeric row per entity (cell line or drug).
  Fingerprints, gene expression, and pathways use this format. It is the
  default and the only format ``MatrixPredictor`` models can consume.
- ``graph`` — one molecular graph object per entity, held in an object array
  instead of a stackable matrix. The ``drugGraph`` featurizer loads
  precomputed PyG graphs (node features ``x``, ``edge_index``); block
  predictors such as DrugGNN run graph convolutions on them.
- ``ragged_sequence`` — one variable-size tensor or sequence per entity, also
  stored as an object array so lengths can differ across drugs. The ``molgnet``
  featurizer exposes MolGNet embeddings this way; block predictors such as
  DIPK consume them without forcing a fixed-width dense matrix.

Cell-line featurizers
~~~~~~~~~~~~~~~~~~~~~

.. include:: _generated_cell_line_featurizers.rst

Drug featurizers
~~~~~~~~~~~~~~~~

.. include:: _generated_drug_featurizers.rst

Predictors
----------

Every predictor inherits exactly one **input interface**. That interface decides whether featurizers are
required and how features are consumed:

- **Feature-free** — the predictor only receives the drug/cell line identifiers, no featurizers are required.
- **Matrix** — the predictor receives a single numeric matrix (can be concatenated from multiple featurizers). This is the case for predictors like ``randomForest``, where it does not matter which omics layer the features originate from.
- **Block** — the predictor receives a dictionary of named featurizer outputs. This is useful for predictors that treat different omics layers separately, e.g. ``molir``, which consumes ``gene_expression``, ``mutations``, and ``copy_number_variation`` as separate blocks. It also allows for predictors to specify that they require certain featurizers to be present in their input data, otherwise they can't work.

.. include:: _generated_predictors.rst

Extensions
----------

DrEvalPy provides a convenient interface to register external components.
This is useful if you want to evaluate a new predictor or featurizer that is not yet part of the catalog.
Details on how to register external components can be found in
:doc:`/python/extensions` for the Python API and :doc:`/cli/extensions`
for the CLI workflow.
