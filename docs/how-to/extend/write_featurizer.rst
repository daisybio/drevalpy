Write a featurizer
==================

.. interface: python

A featurizer turns the views of one side of a dataset (or just the identifiers) into the
numeric, graph or ragged blocks a predictor consumes. This page shows how to write one and
register it. The components in ``docs/examples/`` are imported and checked on every docs
build, so the code shown here keeps working. The toy names they register are listed in
:doc:`/reference/examples`.

Import from ``drevalpy.plugin``
-------------------------------

``drevalpy.plugin`` is the supported import surface for components. It re-exports every
base class, value type and registration decorator you need. Only these names are a
compatibility promise: deeper module paths still import, but a refactor may move them
without a deprecation cycle.

.. code-block:: python

   from drevalpy.plugin import (
       FeatureSource,
       Featurizer,
       FeaturizerResult,
       register_featurizer,
   )

Implement ``fit`` and ``transform``
-----------------------------------

Subclass ``Featurizer`` and register the class with ``register_featurizer(name, *,
description, tags=None, reference=None)``. A featurizer is registered once, whichever sides it
supports. It implements two methods:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Responsibility
   * - ``fit(source, train_ids, *, train_pair_ids=None, es_pair_ids=None)``
     - Learn whatever ``transform`` needs from the training entities. ``train_ids`` are unique;
       ``train_pair_ids`` and ``es_pair_ids`` repeat an entity once per training or
       early-stopping pair, for a featurizer that weights entities by their pairs. Return
       ``self``.
   * - ``transform(source, ids)``
     - Return a ``FeaturizerResult(ids, blocks, projection)``: the entity IDs, a mapping of
       block name to ``FeatureBlock``, and the names of the blocks that make up the flat matrix.
       Keep the order of ``ids``, which may be empty.

``source`` is a ``FeatureSource`` for the featurizer's side. ``source.side`` is ``"sample"`` or
``"drug"``, ``source.view(view, ids)`` reads rows of a view, ``source.feature_names(view)`` its
column names, and ``source.has_view(view)`` tells whether the dataset holds it. ``smiles(ids)``,
``graphs(ids)``, ``tissue(ids)`` and ``uns(key)`` read the SMILES strings, drug graphs, tissue
labels and dataset-level payloads.

Model training calls both methods only with usable entity IDs: it drops the entities whose
input row is entirely NaN (``valid_mask``) beforehand and pads NaN rows back in afterwards. The
featurizer itself never sees them. A fitted featurizer is saved with the model by pickling
it, so keep its fitted state in ordinary attributes.

Declare what it reads and emits
-------------------------------

These class attributes and class methods are read without fitting anything, so a model that
cannot work is rejected when its configuration is built:

``sides``
   The sides the featurizer can be used on: ``frozenset({"sample"})``,
   ``frozenset({"drug"})`` or, the default, both. A model that uses it on another side is
   rejected.
``input_views``
   The views the featurizer reads, such as ``("gene_expression",)``. Leave it empty for a
   featurizer that needs only entity IDs. ``valid_mask`` checks the first of them for all-NaN
   rows.
``output_blocks(config)``
   The blocks the featurizer emits, as ``BlockSpec(name, format)``. The default is one
   ``numeric_matrix`` block named after the configured ``view``, else after the first input
   view. Override it when your blocks are named otherwise or are not numeric. Config validation
   compares these blocks with what the predictor reads.

A featurizer that takes the view from the config declares no ``input_views`` and gives
``__init__`` a ``view`` parameter without a default; the config must then name the view, as in
``myFeaturizer[methylation]``. Such a featurizer overrides ``valid_mask`` to return
``all_nan_rows_mask(source, self._view, ids)``, so the configured view is the one checked for
NaN rows. Its hyperparameters are declared like a predictor's, with
``get_hyperparameter_space`` (see :doc:`write_predictor`), and are passed to ``__init__`` as
keyword arguments.

Registration rejects a class that is not a ``Featurizer``, still has unimplemented abstract
methods, declares malformed ``sides`` or views, or has a hyperparameter entry without a
``default``.

A featurizer that reads a view
------------------------------

This sample featurizer standardizes the ``gene_expression`` view with statistics learned on
the training samples. ``view_matrix`` reads the view with its columns in the order recorded
at fit, and raises ``FeatureMismatchError`` when a fitted feature is missing. ``dense_result``
wraps a matrix as a result with one numeric block.

.. literalinclude:: /examples/toy_sample_featurizer.py
   :language: python
   :caption: docs/examples/toy_sample_featurizer.py

A featurizer that reads only identifiers
----------------------------------------

This drug featurizer reads no view and overrides ``output_blocks``, because its block is not
named after a view. It works against any dataset, including a response-only one.

.. literalinclude:: /examples/toy_drug_featurizer.py
   :language: python
   :caption: docs/examples/toy_drug_featurizer.py

Use one featurizer on both sides
--------------------------------

A featurizer whose logic does not depend on the side keeps the default ``sides`` and reads
``source.side`` if it needs to. The built-in ``identity``, ``constant`` and
``concatFeaturizers`` featurizers work this way.

.. code-block:: python

   from collections.abc import Mapping
   from typing import Any

   import numpy as np

   from drevalpy.plugin import (
       BlockSpec,
       FeatureFormat,
       Featurizer,
       dense_result,
       register_featurizer,
   )


   @register_featurizer("myOnes", description="A constant column per sample or drug.")
   class OnesFeaturizer(Featurizer):
       """Emit one column of ones per entity."""

       @classmethod
       def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
           return (BlockSpec("ones", FeatureFormat.NUMERIC_MATRIX),)

       def fit(self, source, train_ids, *, train_pair_ids=None, es_pair_ids=None):
           return self

       def transform(self, source, ids):
           return dense_result(ids, "ones", np.ones((len(ids), 1)), None)

.. code-block:: python

   from drevalpy.registry.featurizer import names_for_side

   assert "myOnes" in names_for_side("sample")
   assert "myOnes" in names_for_side("drug")

.. _write-featurizer-stored:

Reuse pre-computed features
---------------------------

A featurizer that needs no fitting, whose output for an entity depends only on that entity's
own data and on its settings, can have its output stored in the dataset as a view by
``drevalpy.data.precompute`` (see :doc:`/reference/components`). To take part, subclass
``drevalpy.plugin.PrecomputableFeaturizer``:

- call ``super().__init__(view, settings)`` with the view that holds the output and the
  hyperparameter values that decide it, such as ``{"radius": 2, "n_bits": 2048}``;
- implement ``compute(source, ids)``, which returns one dense row per entity;
- list the views ``compute`` reads in ``source_views``, so ``precompute_all`` skips a dataset
  that lacks them.

The base class implements ``fit``, ``transform`` and ``valid_mask``. It reads the view when the
dataset holds it and either no settings are recorded for it (a view the dataset ships) or the
recorded settings equal its own; otherwise it calls ``compute``. A model with tuned settings
therefore never reads a view that was computed with other ones. The built-in ``fingerprints``,
``chemberta``, ``smilesvec`` and ``bionic`` featurizers work this way.

Next steps
----------

- Check that your featurizers pass the shipped checks: :doc:`test_components`.
- Make them available to the CLI and to other processes: :doc:`load_extensions`.
- Ship them as an installable package: :doc:`publish_plugin`.
- Use them in a model recipe: :doc:`/reference/recipes`.
