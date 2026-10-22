Write a featurizer
==================

.. interface: python

A featurizer turns the raw views of a dataset (or just the identifiers) into the
numeric, graph or ragged payload a predictor consumes. This page shows how to write one,
register it, and bind it to both entity sides when its logic does not depend on the side.
The components in ``docs/examples/`` are imported and checked on every docs build, so the
code shown here keeps working. The toy names they register are listed in
:doc:`/reference/examples`.

Import from ``drevalpy.plugin``
-------------------------------

``drevalpy.plugin`` is the supported import surface for components. It re-exports every
base class, value type and registration decorator you need. Only these names are a
compatibility promise: deeper module paths still import, but a refactor may move them
without a deprecation cycle.

.. code-block:: python

   from drevalpy.plugin import (
       FeatureFormat,
       SampleFeaturizer,
       register_sample_featurizer,
   )

Choose the base class
---------------------

Subclass ``SampleFeaturizer`` or ``DrugFeaturizer`` and register with
``register_sample_featurizer`` or ``register_drug_featurizer``. Two shortcuts cover most cases:

- :ref:`Dense one-view featurizers <write-featurizer-dense>` subclass
  ``DenseViewSampleFeaturizer`` / ``DenseViewDrugFeaturizer`` and override one method.
- :ref:`A featurizer whose logic is the same for samples and drugs <write-featurizer-sides>` is
  written once and bound to both sides with ``register_for_sides``.

Implement the hooks
-------------------

The public ``fit``, ``transform`` and ``transform_blocks`` methods are implemented on the base
class. They find the entities whose feature rows are entirely NaN, run your code on the rest,
and splice NaN rows back into the result. You implement the hooks underneath:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Member
     - Responsibility
   * - ``_fit(source, *, entity_ids=None, ...)``
     - Learn whatever ``_transform_features`` needs from pre-validated entity ids. Return ``self``.
   * - ``_transform_features(source, entity_ids)``
     - Return a ``FeaturizerResult(entity_ids, blocks, projection)``: the entity ids, a mapping of
       block name to ``FeatureBlock``, and the names of the blocks that make up the flat matrix.
       ``FeaturizerResult.matrix`` is a property derived from those, not a constructor argument.
   * - ``_empty_transform_features(source)``
     - Return a zero-row ``FeaturizerResult`` with the fitted output schema.
   * - ``output_dim`` (property)
     - Feature width after fitting.
   * - ``get_state`` / ``set_state``
     - Optional, but required for a fitted featurizer to survive a checkpoint: the checkpoint
       stores the state mapping, not the object. The default implementations are no-ops.

Registration is rejected without two declarations:

**The feature contract.** A ``FeatureFormat`` (``numeric_matrix``, ``graph`` or
``ragged_sequence``) naming the payload this featurizer produces. Composition compares it with
the predictor's ``sample_contract`` or ``drug_contract`` and rejects stacks whose formats
disagree. Set ``contract`` on the class body or pass ``contract=`` to the decorator; when both
are present the decorator wins.

**The input views.** The raw views the featurizer reads, so the data layer never needs a
name-to-view lookup table. Any one of these satisfies the check:

- ``input_views = ("gene_expression",)`` for a fixed set of views.
- ``entity_id_only = True`` for no view at all, just the identifiers.
- ``requires_view = True`` when the view comes from the config as a ``view=`` argument.
- Overriding ``resolve_input_views`` when the views depend on other hyperparameters.

Registration also rejects a class that still has unimplemented abstract methods and names the
missing members, so a forgotten ``_fit`` fails next to its cause.

A featurizer that reads a view
------------------------------

This sample featurizer standardizes the ``gene_expression`` view with statistics learned on
the training samples. It declares ``input_views`` and returns a named numeric block.

.. literalinclude:: /examples/toy_sample_featurizer.py
   :language: python
   :caption: docs/examples/toy_sample_featurizer.py

A featurizer that reads only identifiers
----------------------------------------

This drug featurizer declares its contract on the class body and uses ``entity_id_only``
instead of ``input_views``. It works against any dataset, including a response-only one.

.. literalinclude:: /examples/toy_drug_featurizer.py
   :language: python
   :caption: docs/examples/toy_drug_featurizer.py

.. _write-featurizer-dense:

Compute a dense matrix from one view
------------------------------------

``DenseViewFeaturizer`` implements ``_fit``, ``_transform_features`` and
``_empty_transform_features`` for the shape every dense featurizer shares: use a pre-computed
matrix when the dataset carries one, otherwise compute it from the view, then emit one numeric
block. A subclass overrides only its own step, usually ``_compute_matrix``. Other hooks are
``_fit_state`` (learn state and return the output width), ``_fit_entity_ids``,
``_fetch_hyperparameters``, ``_block_name`` and ``_block_feature_names``.

Subclass the side-bound ``DenseViewSampleFeaturizer`` or ``DenseViewDrugFeaturizer`` for a
one-sided featurizer. Subclass bare ``DenseViewFeaturizer`` when the logic is side-agnostic and
you bind it with ``register_for_sides``.

.. code-block:: python

   import numpy as np

   from drevalpy.plugin import (
       DenseViewSampleFeaturizer,
       FeatureFormat,
       register_sample_featurizer,
   )


   @register_sample_featurizer(
       "logExpression",
       description="log1p of the gene_expression view.",
       contract=FeatureFormat.NUMERIC_MATRIX,
   )
   class LogExpressionFeaturizer(DenseViewSampleFeaturizer):
       """Compress the dynamic range of the gene-expression view."""

       input_views = ("gene_expression",)

       def _compute_matrix(self, source, matrix):
           return np.log1p(np.clip(matrix, 0.0, None))

The class needs no ``output_dim`` or state handling: the base class measures the width while
fitting and restores it from a checkpoint.

.. _write-featurizer-sides:

Bind one implementation to both sides
-------------------------------------

Write a side-agnostic featurizer once and bind it with ``register_for_sides``. Do not write a
sample copy and a drug copy. The registry stamps the entity side onto the class, so one class
registered twice would have its side overwritten; the decorator therefore derives one subclass
per side. The built-in ``identity``, ``constant`` and ``concatFeaturizers`` featurizers are
built this way.

.. code-block:: python

   from typing import ClassVar

   import numpy as np

   from drevalpy.plugin import (
       BlockSpec,
       FeatureFormat,
       Featurizer,
       FeaturizerResult,
       numeric_feature_block,
       register_for_sides,
   )


   @register_for_sides(
       "myOnes",
       description={
           "sample": "A constant column per sample.",
           "drug": "A constant column per drug.",
       },
       contract=FeatureFormat.NUMERIC_MATRIX,
   )
   class SharedOnesFeaturizer(Featurizer):
       """Emit one column of ones per entity."""

       entity_id_only: ClassVar[bool] = True
       output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
           BlockSpec("ones", FeatureFormat.NUMERIC_MATRIX),
       )

       def _fit(self, source, **kwargs):
           return self

       def _transform_features(self, source, entity_ids):
           ones = np.ones((len(entity_ids), 1), dtype=np.float32)
           return FeaturizerResult(entity_ids, {"ones": numeric_feature_block(ones)}, ("ones",))

       def _empty_transform_features(self, source):
           ones = np.empty((0, 1), dtype=np.float32)
           return FeaturizerResult(np.empty(0, dtype=str), {"ones": numeric_feature_block(ones)}, ("ones",))

       @property
       def output_dim(self):
           return 1

``description`` also accepts a plain string used for every side, and ``sides=("sample",)``
binds one side only. The decorator returns the undecorated class, so ``SharedOnesFeaturizer``
stays importable as the shared logic. What lands in the registries are the derived
``SampleOnesFeaturizer`` and ``DrugOnesFeaturizer``, named by stripping a leading ``Shared``
and prefixing the side. Both are injected into your module's namespace, which keeps them
findable when the registries are rebuilt.

.. code-block:: python

   from drevalpy.registry import drug_featurizer, sample_featurizer

   assert "myOnes" in sample_featurizer.list()
   assert "myOnes" in drug_featurizer.list()

Reuse pre-computed features
---------------------------

``fetch``, ``fetch_precomputed``, ``store`` and ``list_stored_variants`` come from
``FeaturizerStorageMixin``, which ``Featurizer`` already includes, so every featurizer has them.
Name the mixin only to annotate against it, or to override ``_fetch_by_key`` and
``_store_by_key`` when you need storage of your own.

Next steps
----------

- Check that your featurizers conform to the contract: :doc:`test_components`.
- Make them available to the CLI and to other processes: :doc:`load_extensions`.
- Ship them as an installable package: :doc:`publish_plugin`.
- Use them in a model recipe: :doc:`/reference/recipes`.
