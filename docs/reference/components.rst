Components
==========

A model is built from up to three components: a sample featurizer, a drug featurizer and a
predictor. Featurizers and predictors are looked up by name in their registries. A featurizer is
registered once and declares the sides it can be used on, sample, drug or both.
``drevalpy list sample-featurizers``, ``drevalpy list drug-featurizers`` and
``drevalpy list predictors`` print the registered components, including those added by
extensions. The tables below list the built-in ones. :doc:`recipes` describes how components are
combined into a model.

Featurizer formats
------------------

The output format of a featurizer is one of the values of ``FeatureFormat``.

``numeric_matrix``
   One dense numeric row per entity, a sample or a drug. Fingerprints, gene expression and
   pathways use this format.

``graph``
   One molecular graph per entity, held in an object array. The ``drugGraph`` featurizer reads
   stored graphs with node features ``x`` and ``edge_index``, or computes them from the SMILES.

``ragged_sequence``
   One variable-size tensor or sequence per entity, held in an object array so that lengths can
   differ. The ``molgnet`` featurizer returns its embeddings in this format.

A featurizer declares the blocks it emits and their formats, and a predictor what it reads from
each side. A model whose featurizer output does not match is rejected when the configuration is
validated.

Precomputable featurizers
-------------------------

A featurizer marked precomputable depends only on the entity (the sample or the drug), not on
the training labels or the fold. Its output can therefore be computed once and stored in the
dataset file. ``precompute(dataset, featurizer_name, side, hyperparameters, view=None)`` stores
the output for a list of hyperparameter configurations, or for a given number of configurations
drawn from the search space (seeded, so the draw is reproducible), together with the default
configuration, and returns the keys it stored. ``precompute_all(dataset, n_variants=10)`` does
this for every precomputable featurizer, on each side it supports, whose input data the dataset
has. Both live in ``drevalpy.data.precompute``.

The stored variants are listed under ``sample_featurizer_variants`` and
``drug_featurizer_variants`` in ``uns``; see :doc:`datasets`. When a model is trained or
predicts, a featurizer whose stored variant matches its hyperparameters uses that variant and is
neither fitted nor applied.

Featurizers
-----------

.. include:: _generated_featurizers.rst

Predictors
----------

Every predictor declares what it reads from each side, ``sample_inputs`` and ``drug_inputs``.
The declaration decides whether a featurizer is required on that side and how the predictor
receives its output.

None
   The predictor reads no features on that side and forbids a featurizer there. A recipe may
   consist of a predictor that reads none on both sides alone, for example ``naiveMean``. The
   single-drug predictors read none on the drug side: they fit one model per drug and route
   pairs by drug identifier.

Matrix
   The predictor receives a single numeric matrix, which may be the concatenation of several
   featurizers. It does not matter which view a column came from. Predictors such as
   ``randomForest`` and ``elasticNet`` are of this kind, and they take only ``numeric_matrix``
   features.

Blocks
   The predictor receives a dictionary of named feature blocks and declares the blocks it
   requires; "or" separates alternatives of which one must be present. For example, ``molir``
   requires the sample blocks ``gene_expression``, ``mutations`` and
   ``copy_number_variation_gistic``.

.. include:: _generated_predictors.rst
