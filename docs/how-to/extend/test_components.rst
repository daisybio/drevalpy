Test your components
====================

.. interface: python

``drevalpy.testing`` ships in the wheel so that a plugin's own test suite can import it.
Together with ``drevalpy.synthetic``, it removes the two things that otherwise block an offline CI job for a plugin: every registered
dataset lives in a credentialed bucket, and a predictor needs a fully featurized batch.

What ``drevalpy.synthetic`` and ``drevalpy.testing`` provide
------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Helper
     - What it gives you
   * - ``drevalpy.synthetic(...)``
     - An in-memory ``Dataset`` (60 samples, 8 drugs and 6 tissues by default). It holds the
       ``gene_expression`` view by default; pass ``omics=[...]`` for the sample views your
       featurizer reads, or ``omics=()`` for a response-only dataset.
   * - ``build_synthetic_batch(dataset, sample_block_names=(), drug_block_names=(), ...)``
     - A ``ModelInputBatch`` with drawn features and a learnable response, so a predictor is
       testable without composing a model.
   * - ``observed_pairs(dataset)``
     - Every measured sample-drug pair, as a ``ResponseBatch``.
   * - ``FEATURIZER_CHECKS`` / ``PREDICTOR_CHECKS``
     - A tuple of three conformance checks each, ready to parametrize over. The third check of
       each is a pickle round trip (``check_featurizer_pickle_round_trip``,
       ``check_predictor_pickle_round_trip``).
   * - ``check_plugin(name)``
     - Assert that an installed plugin's entry point is declared and loads, and that its
       components resolve through the registries.

Check a featurizer
------------------

Each check takes ``(cls, fixture, **kwargs)``. For featurizers the fixture is a dataset. The
three checks verify that the component instantiates, that ``transform`` emits the blocks
``output_blocks`` declares, and that a fitted instance dumped and loaded with joblib reproduces
the original's output. The last one catches the expensive bug: a saved model pickles its fitted
components, so a fitted attribute that does not pickle breaks a reloaded model. A failing check
raises ``ConformanceError``, which subclasses ``AssertionError``.

.. code-block:: python

   from typing import ClassVar

   import numpy as np

   from drevalpy import synthetic
   from drevalpy.plugin import Featurizer, Side, dense_result, register_featurizer, view_matrix
   from drevalpy.testing import FEATURIZER_CHECKS


   @register_featurizer("logExpression", side=Side.SAMPLE, description="log1p of the gene_expression view.")
   class LogExpressionFeaturizer(Featurizer):
       input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)

       def fit(self, source, train_ids, *, train_pair_ids=None, es_pair_ids=None):
           names = source.feature_names("gene_expression")
           self._names = None if names is None else tuple(names)
           return self

       def transform(self, source, ids):
           matrix = view_matrix(source, "gene_expression", ids, self._names)
           return dense_result(ids, "gene_expression", np.log1p(np.clip(matrix, 0.0, None)), self._names)


   dataset = synthetic(omics=["gene_expression"])
   for check in FEATURIZER_CHECKS:
       check(LogExpressionFeaturizer, dataset)

Pass ``omics=[...]`` with every view your featurizer lists in ``input_views``. A featurizer that
reads no view works against a response-only dataset (``omics=()``) too.

Check a predictor
-----------------

For predictors the fixture is a batch. Ask ``build_synthetic_batch`` for the sample and drug
blocks the predictor requires by name; a matrix predictor only needs some features. The third
check trains the predictor, pickles it and requires the loaded copy to predict the same values.

.. code-block:: python

   from drevalpy.plugin import FeatureFreePredictor, register_predictor
   from drevalpy.testing import PREDICTOR_CHECKS, build_synthetic_batch


   @register_predictor("meanBaseline", description="Predict the mean training response.")
   class MeanBaseline(FeatureFreePredictor):
       def _fit(self, batch):
           self._mean = float(np.mean(batch.response))

       def _predict(self, batch):
           return np.full(batch.n_pairs, self._mean)


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

   folds = splitter.get("LSO")(synthetic(), n_splits=3, quality=CurveQuality.default())
   assert len(folds) == 3

Plots have no conformance check either: call ``sections(result)`` on a result and save each
figure (see :doc:`write_visualization`).

Put the checks in your test suite
---------------------------------

Parametrize over the check tuples so one test function covers every check. This module imports
the reader's own package, so it is shown but not run here.

.. snippet: skip

.. code-block:: python

   import pytest

   from drevalpy import synthetic
   from drevalpy.testing import FEATURIZER_CHECKS

   from my_plugin.featurizers import MyFeaturizer


   @pytest.mark.parametrize("check", FEATURIZER_CHECKS)
   def test_my_featurizer_conforms(check):
       check(MyFeaturizer, synthetic(omics=["gene_expression"]))

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
