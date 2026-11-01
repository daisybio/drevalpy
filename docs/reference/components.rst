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

A featurizer marked precomputable needs no fitting: its output for an entity (a sample or a
drug) depends only on that entity's own data and on the featurizer's settings, not on the
training entities or the fold. Its output can therefore be computed once and stored in the
dataset as a view. ``precompute(dataset, featurizer_name, side)`` computes the output at the
featurizer's default settings, stores it as the featurizer's view (a ``response.varm`` key on
the drug side, a modality on the sample side) and records the settings under
``precomputed_views`` in ``uns`` (see :doc:`datasets`). It returns the view it wrote, or
``None`` when the dataset already held a view the featurizer reads. ``precompute_all(dataset)``
does this for every precomputable featurizer, on each side it supports, whose source views the
dataset has, and returns the views written. Both live in ``drevalpy.data.precompute``.

When a model is trained or predicts, a precomputable featurizer reads its view when the dataset
holds it and either no settings are recorded for it, as for the views a dataset ships, or the
recorded settings equal its own. Otherwise it computes its output, so a tuned setting such as
``n_bits`` of ``fingerprints`` never reads a view computed with another value. Only the default
settings are stored; there is one view per featurizer.

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
