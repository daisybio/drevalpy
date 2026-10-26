How models are built from components
====================================

In DrEvalPy a model is not a single class that you subclass for every new
idea. It is an assembly of *components*, and the assembly is described by data.
This page explains the parts, why they are separate, and how the rules for
combining them are enforced. The commands that build and run models are in the
how-to guides: :doc:`/how-to/choose_models` and :doc:`/how-to/extend/index`.

Three kinds of part
-------------------

A model has up to three components:

**Sample featurizer.**
   Turns one or more views of the samples, such as ``gene_expression``, into
   numbers a predictor can use.

**Drug featurizer.**
   Does the same for drugs, for example Morgan fingerprints, or a molecular
   graph.

**Predictor.**
   Learns the mapping from the sample and drug features to the response, and
   predicts it for new pairs.

Splitting a model this way means that a new gene expression preprocessing step
can be combined with every existing predictor, and a new predictor immediately
works with every existing featurizer that produces compatible features. Without
the split, each combination would be its own class. With it, ElasticNet,
RandomForest, XGBoost and LightGBM share the same featurizers, and the same
featurizer can be compared across predictors. The cost is that compatibility has
to be checked, which the next sections describe.

A *view* is a named table of features on one side, such as ``gene_expression``
for samples or ``fingerprints`` for drugs. Featurizers read views. Some featurizers
are bound to one view, while a few general ones, such as ``raw`` and ``pca``, take
the view as a parameter.

Featurizers versus encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A *featurizer* is a component fitted without looking at the response. It learns
something about the training entities only, for example a scaling or a principal
component basis, and it can often be computed once and reused. An *encoder* is
different: it is a neural network layer that is trained jointly with the
predictor, by backpropagating the prediction loss. Encoders are part of the
predictor, not a separate component. This is why literature models such as
DrugGNN appear as one predictor with a featurizer that supplies its inputs.

Feature formats and predictor interfaces
----------------------------------------

Features come in three formats: a numeric matrix, a graph, or a ragged sequence
(sequences of different lengths, such as tokenized strings). Each featurizer
declares which format it produces, and each predictor declares the interface it
consumes:

- A *feature-free* predictor uses no features at all and forbids featurizers.
  The overall-mean baseline is one. Other naive baselines use only entity
  identities, and for tissue means a tissue label.
- A *matrix* predictor takes a numeric matrix, and so accepts the usual
  scikit-learn style models.
- A *block* predictor takes named feature blocks and declares which ones it
  requires, so that a neural network can consume a graph for the drug and a
  matrix for the sample.

Features that a featurizer produces are bundled as a result containing the
entity identifiers, a set of named blocks and a projection, the ordered list of
blocks that make up the featurizer's default matrix. The matrix is derived from
the blocks when it is meaningful. An empty projection gives a matrix with no
columns, one block passes through unchanged, and several blocks are joined only
if all of them are numeric and aligned by entity. Graph and sequence blocks stay
addressable by name instead of being flattened.

Describing a model
------------------

There are three equivalent ways to say which components a model uses, and they
trade conciseness against expressiveness:

- A *recipe string* is one line, with the components separated by colons. The
  grammar is ``predictor``, ``sample:predictor`` or ``sample:drug:predictor``.
  For example, ``scaledGeneExpression:fingerprints:elasticNet`` scales gene
  expression, computes drug fingerprints and fits an elastic net. A plus sign joins several
  featurizers on one side, and square brackets name the view a flexible featurizer should read.
  Recipes carry no hyperparameter space, so each component falls back to its built-in one.
- A *model file* (YAML) says the same in more words and can also declare
  hyperparameter spaces for individual components.
- A *model configuration* in Python holds the same information as the file, as
  objects.

All three are parsed into one internal form, so a model behaves the same
whichever notation you chose. The grammar, field names and examples are in
:doc:`/reference/recipes`. Named, ready-to-run models are called zoo presets and
are looked up by their zoo name; see :doc:`/reference/model_zoo`. A model
specification pairs a name with a configuration, and it holds no fitted state, so
one specification can be reused across folds and tuning trials (see
:doc:`training_lifecycle`).

A recipe is also a model name. The command line and ``ModelSpec.from_name`` accept a recipe
wherever they accept a zoo name, and the model is then reported under the recipe string,
views included, so every combination has a unique name. A zoo name wins when both readings are
possible. A name that is neither raises ``ValueError``.

Component grids
~~~~~~~~~~~~~~~

Because components combine freely, comparing featurizers or predictors means evaluating a grid of
recipes: every sample featurizer with every drug featurizer and every predictor. Most products of
a list are not all valid, since a tissue-mean predictor cannot use gene expression. Rather than
making the user work out which combinations fit, a grid applies the validation below to each
product and keeps the valid ones. The invalid ones are reported with the reason instead of
failing the whole experiment. The commands are in :doc:`/how-to/evaluate_component_grids`.

Single-drug models
~~~~~~~~~~~~~~~~~~

Some predictors fit one separate estimator per drug, for example
``singleDrugElasticNet``. Per-drug routing needs the drug identity, so when a
recipe omits the drug featurizer, the configuration adds the ``identity``
featurizer, which one-hot encodes drug identifiers, and the predictor reports
that its scope is a single drug. MOLIR is a single-drug literature model too.

Validation happens before training
----------------------------------

A recipe that cannot work should fail when it is written, not hours into an
experiment. When a configuration is resolved, DrEvalPy checks that:

- a predictor that uses no features has no featurizers configured, and one that
  does use features has the featurizers it needs;
- each featurizer's output format is accepted by the predictor;
- the feature blocks that a block predictor requires are produced by the
  featurizers, with the expected schema;
- the blocks that the predictor names in ``required_sample_blocks`` and
  ``required_drug_blocks``, such as the ``tissue`` block of a tissue-mean predictor, are among
  the blocks the featurizers emit;
- the prediction mode of the model is one the predictor supports;
- a single-drug predictor is paired with the drug identity featurizer.

For example, a matrix predictor rejects a graph feature, a feature-free
predictor rejects any featurizer, and a tissue-mean predictor rejects a gene expression
featurizer. The error names the component and the format or block that failed to match. Because a model is a plain description of components,
these checks are ordinary code that runs over the description, and this is the
property that makes it safe to compose models freely.

Hyperparameters of a composed model
-----------------------------------

Each component owns its hyperparameters and its default search space. When a
model is tuned, the spaces of all components are merged into one. To keep the
parameters of different components apart, every key is qualified by the part it
belongs to: the predictor, the sample featurizer or the drug featurizer, then the
component name, then the parameter. A featurizer that reads a view carries the
view in its name, so the same featurizer used on two views gives two independent
sets of parameters. A short parameter name is accepted when it is unambiguous
across the model, and an ambiguous one is rejected instead of being applied to
every component that has it. The full key grammar is in
:doc:`/reference/hyperparameters`.

A hyperparameter space declared in a file or configuration replaces the built-in
space of that component. It does not extend it.

Why it matters for results
--------------------------

The same composition describes the literature models. DrugGNN, for example, is
a graph drug featurizer, a landmark-gene expression sample featurizer and a
neural predictor.
Because every model is built the same way, a benchmark compares predictors and
feature choices under identical splits, fitting scopes and scoring, and a
difference in results can be traced to a difference in components. How components
are registered and discovered is in :doc:`registries_and_plugins`.
