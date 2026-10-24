Test your components
====================

.. interface: python

``drevalpy.testing`` ships in the wheel so that a plugin's own test suite can import it. It
removes the two things that otherwise block an offline CI job for a plugin: every registered
dataset lives in a credentialed bucket, and a predictor needs a fully featurized batch.

What ``drevalpy.testing`` provides
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Helper
     - What it gives you
   * - ``build_synthetic_dataset(...)``
     - An in-memory ``Dataset`` (60 samples, 8 drugs and 6 tissues by default). It is
       response-only unless you pass ``omics=[...]`` to add the sample modalities your featurizer
       reads. It is an alias of ``drevalpy.synthetic``.
   * - ``build_synthetic_batch(dataset, sample_block_names=(), drug_block_names=(), ...)``
     - A ``ModelInputBatch`` with drawn features and a learnable response, so a predictor is
       testable without composing a model.
   * - ``observed_pairs(dataset)``
     - Every measured sample-drug pair, as a ``ResponseBatch``.
   * - ``FEATURIZER_CHECKS`` / ``PREDICTOR_CHECKS``
     - A tuple of three conformance checks each, ready to parametrize over.
   * - ``check_plugin(name)``
     - Assert that an installed plugin's entry point is declared and loads, and that its
       components resolve through the registries.

Check a featurizer
------------------

Each check takes ``(cls, fixture, **kwargs)``. For featurizers the fixture is a dataset. The
three checks verify that the component instantiates, that ``output_dim`` agrees with the width
``transform`` actually produced, and that a fresh instance restored from ``get_state``
reproduces the original's output. The last one catches the expensive bug: a fitted attribute
left out of ``get_state`` makes a reloaded checkpoint silently predict something else. A
failing check raises ``ConformanceError``, which subclasses ``AssertionError``.

.. code-block:: python

   import numpy as np

   from drevalpy.plugin import (
       DenseViewSampleFeaturizer,
       FeatureFormat,
       register_sample_featurizer,
   )
   from drevalpy.testing import FEATURIZER_CHECKS, build_synthetic_dataset


   @register_sample_featurizer(
       "logExpression",
       description="log1p of the gene_expression view.",
       contract=FeatureFormat.NUMERIC_MATRIX,
   )
   class LogExpressionFeaturizer(DenseViewSampleFeaturizer):
       input_views = ("gene_expression",)

       def _compute_matrix(self, source, matrix):
           return np.log1p(np.clip(matrix, 0.0, None))


   dataset = build_synthetic_dataset(omics=["gene_expression"])
   for check in FEATURIZER_CHECKS:
       check(LogExpressionFeaturizer, dataset)

Pass ``omics=[...]`` with every view your featurizer lists in ``input_views``. A featurizer that
reads no view works against the default response-only dataset.

Check a predictor
-----------------

For predictors the fixture is a batch. Ask ``build_synthetic_batch`` for the sample and drug
blocks the predictor requires by name; a matrix predictor only needs some features.

.. code-block:: python

   from drevalpy.plugin import FeatureFormat, FeatureFreePredictor, register_predictor
   from drevalpy.testing import PREDICTOR_CHECKS, build_synthetic_batch


   @register_predictor(
       "meanBaseline",
       description="Predict the mean training response.",
       sample_contract=FeatureFormat.NUMERIC_MATRIX,
       drug_contract=FeatureFormat.NUMERIC_MATRIX,
   )
   class MeanBaseline(FeatureFreePredictor):
       def _fit(self, batch):
           self._mean = float(np.mean(batch.response))

       def _predict(self, batch):
           return np.full(batch.n_pairs, self._mean)

       def get_state(self):
           return {"mean": self._mean} if hasattr(self, "_mean") else {}

       def set_state(self, state):
           if "mean" in state:
               self._mean = state["mean"]


   batch = build_synthetic_batch(dataset, sample_block_names=("gene_expression",))
   for check in PREDICTOR_CHECKS:
       check(MeanBaseline, batch)

Check a splitter
----------------

A splitter has no conformance check; resolve your mode through the registry instead of calling
the function directly. The registry wraps every splitter in its leakage check, so a splitter
that leaks a sample (for ``LSO``) into both train and test raises ``SplitValidationError``.
This block uses the built-in ``LSO``; substitute your own mode name.

.. code-block:: python

   from drevalpy.plugin import CurveQuality
   from drevalpy.registry import splitter

   folds = splitter.get("LSO")(build_synthetic_dataset(), n_splits=3, quality=CurveQuality.default())
   assert len(folds) == 3

Visualizations are checked the same manual way: call ``compute`` first, then the renderers (see
:doc:`write_visualization`).

Put the checks in your test suite
---------------------------------

Parametrize over the check tuples so one test function covers every check. This module imports
the reader's own package, so it is shown but not run here.

.. snippet: skip

.. code-block:: python

   import pytest

   from drevalpy.testing import FEATURIZER_CHECKS, build_synthetic_dataset

   from my_plugin.featurizers import MyFeaturizer


   @pytest.mark.parametrize("check", FEATURIZER_CHECKS)
   def test_my_featurizer_conforms(check):
       check(MyFeaturizer, build_synthetic_dataset(omics=["gene_expression"]))

Check the installed plugin
--------------------------

``check_plugin(name)`` asserts that the entry point called ``name`` is declared, that its
module imports, and that at least one component registered under its package. It raises
``PluginCheckError`` otherwise and returns a ``PluginReport`` naming the components. It needs the
plugin to be installed, so it belongs in the plugin's CI rather than here.

.. snippet: skip

.. code-block:: python

   from drevalpy.testing import check_plugin


   def test_plugin_is_wired_up():
       report = check_plugin("my_plugin")
       assert report.components

:doc:`publish_plugin` shows how to declare the entry point.
