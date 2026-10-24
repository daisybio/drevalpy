Components
==========

A model is built from up to three components: a sample featurizer, a drug featurizer and a
predictor. Each is looked up by name in its own registry. ``drevalpy list sample-featurizers``,
``drevalpy list drug-featurizers`` and ``drevalpy list predictors`` print the registered
components, including those added by extensions. The tables below list the built-in ones.
:doc:`recipes` describes how components are combined into a model.

Featurizer formats
------------------

The output format of a featurizer is one of the values of ``FeatureFormat``.

``numeric_matrix``
   One dense numeric row per entity, a sample or a drug. Fingerprints, gene expression and
   pathways use this format.

``graph``
   One molecular graph per entity, held in an object array. The ``drugGraph`` featurizer loads
   precomputed graphs with node features ``x`` and ``edge_index``.

``ragged_sequence``
   One variable-size tensor or sequence per entity, held in an object array so that lengths can
   differ. The ``molgnet`` featurizer returns its embeddings in this format.

A predictor declares the format it accepts for samples and for drugs. A model whose featurizer
output does not match is rejected when the configuration is validated.

Precomputable featurizers
-------------------------

A featurizer marked precomputable depends only on the entity (the sample or the drug), not on
the training labels or the fold. Its output can therefore be computed once and stored in the
dataset file. ``Dataset.precompute(featurizer_cls, hyperparameters, view=None)`` stores the
output for a list of hyperparameter configurations, or for a given number of configurations
drawn from the search space, together with the default configuration.
``Dataset.precompute_all(n_variants=10)`` does this for every precomputable featurizer whose
source data the dataset has.

The stored variants are listed under ``sample_featurizer_variants`` and
``drug_featurizer_variants`` in ``uns``; see :doc:`datasets`. When a featurizer is fitted, it
uses a stored variant whose hyperparameters match.

Sample featurizers
------------------

.. include:: _generated_sample_featurizers.rst

Drug featurizers
----------------

.. include:: _generated_drug_featurizers.rst

Predictors
----------

Every predictor has one of three input interfaces. The interface decides whether featurizers
are required and how the predictor receives their output.

Feature-free
   The predictor receives only the sample and drug identifiers. It needs no featurizer, and a
   recipe may consist of the predictor alone, for example ``naiveMean``.

Matrix
   The predictor receives a single numeric matrix, which may be the concatenation of several
   featurizers. It does not matter which view a column came from. Predictors such as
   ``randomForest`` and ``elasticNet`` are of this kind, and they take only ``numeric_matrix``
   features.

Block
   The predictor receives a dictionary of named feature blocks and declares the blocks it
   requires. For example, ``molir`` requires the sample blocks ``gene_expression``,
   ``mutations`` and ``copy_number_variation_gistic``, and is a single-drug predictor, which
   trains one model per drug.

.. include:: _generated_predictors.rst
