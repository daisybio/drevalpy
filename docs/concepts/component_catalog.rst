Component catalog
=================

This catalog is the vocabulary of a DrEvalPy model. Every model is assembled
from components with three distinct roles:

- a **cell-line featurizer** represents the biological sample,
- a **drug featurizer** represents the compound, and
- a **predictor** maps those representations to a drug-response estimate.

The names below are the stable registry names used in recipes and model-zoo
definitions. They are case-sensitive. At this stage, focus on what each
component contributes; the next page, :doc:`from_components_to_models`,
explains how the names fit together and how compatibility is checked.

Featurizer outputs
------------------

Every featurizer can expose the same fitted representation in two ways:

- as a **matrix** (``transform``) — one row per entity, used when a
  ``MatrixPredictor`` builds a single pair-level design matrix;
- as a **dict of named blocks** (``transform_blocks``) — one or more
  arrays keyed by block name (for example ``pathways`` or ``gene_expression``),
  used by a ``BlockPredictor`` that keeps side-specific or named tensors
  separate.

By default, ``transform_blocks`` wraps the matrix under a single
``default`` key. Featurizers that preserve view or modality identity
override that method so block predictors receive the named arrays they
declare. Feature format (numeric matrix, graph, ragged sequence) applies
to the payload type inside either form; it does not replace this
matrix-versus-dict distinction.

Cell-line featurizers
---------------------

.. include:: _generated_cell_line_featurizers.rst

Drug featurizers
----------------

.. include:: _generated_drug_featurizers.rst

Predictors
----------

Every predictor inherits exactly one **input interface**. That interface decides whether featurizers are
required and how features are consumed:

- **Feature-free** — response / pair identifiers only; no featurizers
- **Matrix** — one numeric pair-level design matrix from configured featurizers
- **Block** — side-specific or named featurizer blocks (including side-specific
  matrices); not a single flattened design matrix

Feature **format** (``numeric_matrix``, ``graph``, ``ragged_sequence``) is
orthogonal: matrix predictors reject graph/ragged payloads, while a future
composed graph model would be block-based with a graph drug format. Neural
encoders stay private inside predictors.

.. include:: _generated_predictors.rst

Naive baselines carry the discovery tag ``baseline``. Literature ports attach
a structured ``LiteratureReference``. Neither tags nor references change
validation or execution.

``drugGraph`` provides the graph block consumed by the DrugGNN zoo preset.

From catalog to composition
---------------------------

Matrix and block models still take one row from each table — for example
``scaledGeneExpression:fingerprints:elasticNet``. Feature-free predictors are
**predictor-only** (no featurizer slots).

The next page turns those names into recipes, omics-view selectors, multi-view
concatenation, compatibility checks, and component hyperparameters. Continue
with :doc:`from_components_to_models`; the remaining sections on this page are
reference notes for extensions and older interfaces.

Extensions
----------

External components and optional zoo files can be registered before models are
constructed (via ``load_extensions`` in the Python API). See
:doc:`/python/custom_models` for a full registration example and
:doc:`/python/architecture` for contracts and composition rules.

Backward compatibility
----------------------

Registry names
~~~~~~~~~~~~~~

Before 1.6.0, many of these atoms were reached only through factory dict
entries or deep model modules. Registry string names
(``elasticNet``, ``fingerprints``, …) are the stable composition surface.
Factory dictionaries remain available for backward compatibility, but are
deprecated and may be removed in a future release — prefer
``construct_model`` and the names in the tables above.
