Write a predictor
=================

.. interface: python

A predictor is the model proper: it learns from a batch of sample-drug pairs and scores new
pairs. This page shows how to write one for each of the three input interfaces and how to
declare its hyperparameters, state and provenance. For the featurizers that feed it, see
:doc:`write_featurizer`.

Pick the input interface
------------------------

A predictor inherits **exactly one** of three bases from ``drevalpy.plugin`` and registers with
``register_predictor``:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Base
     - The predictor sees
     - You implement
   * - ``FeatureFreePredictor``
     - Pair identifiers and responses only.
     - ``_fit(batch)``, ``_predict(batch)``
   * - ``MatrixPredictor``
     - One dense pair-level design matrix.
     - ``_fit_matrix(x, y)``, ``_predict_matrix(x)``
   * - ``BlockPredictor``
     - The named featurizer blocks of each side, kept apart.
     - ``_fit(batch)``, ``_predict(batch)``

The public ``fit`` and ``predict`` are implemented on the base class. ``fit`` raises
``ValueError`` for a batch without responses and drops pairs whose features are NaN;
``predict`` returns NaN for those pairs. Your hooks never see them.

Every predictor declares ``sample_input_format`` and ``drug_input_format``, each a
``FeatureFormat``. Pass them to the decorator or set them on the class body; the decorator wins
when both are given. A missing or wrong-typed format raises ``TypeError``. ``MatrixPredictor``
requires ``numeric_matrix`` for both. A feature-free predictor still needs both formats because
the composition checker compares them before it knows the interface, but composition then
forbids sample and drug featurizers for it.

.. tab-set::

   .. tab-item:: Feature-free

      The simplest case: predict the training mean for every pair.

      .. literalinclude:: /examples/toy_mean_predictor.py
         :language: python
         :caption: docs/examples/toy_mean_predictor.py

   .. tab-item:: Matrix

      ``MatrixPredictor`` calls ``batch.to_feature_matrix()`` for you, so you work on a plain
      array. This example wraps scikit-learn's ridge regression and also declares a
      hyperparameter space and a literature reference.

      .. literalinclude:: /examples/toy_ridge_predictor.py
         :language: python
         :caption: docs/examples/toy_ridge_predictor.py

   .. tab-item:: Block

      ``BlockPredictor`` reads named blocks from ``batch.sample`` and ``batch.drug`` (each a
      ``FeaturizerResult``) instead of one flattened matrix. Blocks are indexed by entity, so
      expand them to one row per pair with ``batch.sample_pair_idx`` and
      ``batch.drug_pair_idx``, as the example does.

      The input formats constrain the **format** of each side. To also require named blocks, set
      ``required_sample_block_specs`` or ``required_drug_block_specs`` to a tuple of
      ``BlockSpec(name, format)``. Composition rejects a recipe whose featurizer does not emit
      them. ``required_sample_block_alternatives`` accepts any one of several blocks. The
      attributes ``required_sample_blocks`` and ``required_drug_blocks`` list block names only,
      and are enforced too: a recipe whose featurizer does not emit a named block is rejected
      when the configuration is built, not when the model trains. A grid skips such a
      combination and reports the reason; see :doc:`/how-to/evaluate_component_grids`.

      .. literalinclude:: /examples/toy_block_predictor.py
         :language: python
         :caption: docs/examples/toy_block_predictor.py

Write a matrix predictor in a few lines
---------------------------------------

This predictor needs no scikit-learn. It solves a penalised least-squares problem on the design
matrix and declares one tunable hyperparameter.

.. code-block:: python

   import numpy as np

   from drevalpy.plugin import FeatureFormat, MatrixPredictor, register_predictor


   @register_predictor(
       "leastSquares",
       description="Penalised least squares on the design matrix.",
       sample_input_format=FeatureFormat.NUMERIC_MATRIX,
       drug_input_format=FeatureFormat.NUMERIC_MATRIX,
   )
   class LeastSquaresPredictor(MatrixPredictor):
       """Fit weights with the normal equations."""

       @classmethod
       def get_hyperparameter_space(cls):
           return {"penalty": {"type": "float", "low": 1e-6, "high": 10.0, "log": True, "default": 1e-3}}

       def _fit_matrix(self, x, y):
           penalty = float(self._hyperparameters["penalty"])
           design = np.hstack([x, np.ones((len(x), 1))])
           gram = design.T @ design + penalty * np.eye(design.shape[1])
           self._weights = np.linalg.solve(gram, design.T @ y)

       def _predict_matrix(self, x):
           return np.hstack([x, np.ones((len(x), 1))]) @ self._weights

       def get_state(self):
           return {"weights": self._weights} if hasattr(self, "_weights") else {}

       def set_state(self, state):
           if "weights" in state:
               self._weights = state["weights"]

       def is_fitted(self):
           return hasattr(self, "_weights")

Declare hyperparameters
-----------------------

Override the class method ``get_hyperparameter_space`` to declare what is tunable. Every entry
needs a ``default``: that value is used when nothing is tuned, and registration rejects a space
without one. A float entry also takes ``low``, ``high`` and ``log``, as in both examples above.
The hyperparameters in effect are available as ``self._hyperparameters``, with your overrides
merged onto the defaults.

Persist the fitted state
------------------------

A checkpoint stores the mapping from ``get_state`` and hands it to ``set_state`` on a fresh
instance. Both default to no-ops, and ``is_fitted()`` is true when ``get_state()`` is not empty.
Include everything ``_predict`` reads. A fitted attribute left out of ``get_state`` makes a
reloaded checkpoint silently predict something else; :doc:`test_components` shows the check
that catches it.

.. warning::

   Checkpoints are serialized with joblib and pickle. Load only checkpoints you trust.

Credit the source of a ported model
-----------------------------------

``LiteratureReference`` is optional provenance metadata for components ported from a paper or
repository. Pass it as ``reference=`` to the register decorator, as ``toyRidge`` does above. It
does not change training, composition or checkpoints. ``repo_url`` is required;
``citation_text``, ``citation_doi`` and ``deviations`` are optional strings.

.. code-block:: python

   from drevalpy.plugin import LiteratureReference

   reference = LiteratureReference(
       repo_url="https://github.com/example/original-model",
       citation_text="Example et al., 2024.",
       deviations="Uses ridge regression instead of the original network.",
   )

Fit one model per drug
----------------------

A single-drug predictor fits one model per drug. With ``--n-jobs`` (``n_jobs`` in Python) above 1
it fits them in parallel worker processes, one drug per worker, and
``batch.training_context.n_jobs`` tells a predictor how many it may use. The helper
``run_per_drug`` in ``drevalpy.components.predictors.single_drug_routing`` does the fan-out for
the built-in single-drug predictors: it seeds each drug from the run seed and the drug id, so the
fitted models do not depend on ``n_jobs``. If you write a predictor that fits per drug, call it
with the per-drug fit function, the tasks (one per drug) and ``n_jobs``, and make sure that the
fit works inside a worker:

- **Do not start processes yourself.** Nested workers multiply the number of processes beyond
  ``n_jobs``. Use ``num_workers=0`` for a PyTorch ``DataLoader``, and leave the thread count of
  scikit-learn and PyTorch alone: the worker already limits it to one.
- **Do not write to a fixed path that the drugs share.** Write checkpoints below
  ``batch.training_context.checkpoint_dir``, the directory that ``DRPModel.train`` passes in, and
  in a subdirectory of your own for each drug. Turn off the default logger of PyTorch Lightning, which
  writes to ``lightning_logs/`` in the working directory and makes concurrent fits race for the
  same version directory. The built-in neural predictors do it with
  ``default_logger=False`` in their ``LightningRun``; Weights & Biases logging stays on when a
  tracking run is active.
- **Return the fitted model instead of storing it on ``self``.** A worker fits a copy of the
  predictor, so an attribute that the fit function sets is lost. Collect the returned models in
  the parent and keep them in the state that ``get_state`` returns.

Check that it works
-------------------

Run the shipped conformance checks on your predictor with a batch from ``drevalpy.testing``,
then use it in a model. The first block needs no dataset files; the second trains the
feature-free baseline defined below through the Python API.

.. code-block:: python

   from drevalpy.testing import PREDICTOR_CHECKS, build_synthetic_batch, build_synthetic_dataset

   dataset = build_synthetic_dataset(omics=["gene_expression"])
   batch = build_synthetic_batch(dataset, sample_block_names=("gene_expression",))
   for check in PREDICTOR_CHECKS:
       check(LeastSquaresPredictor, batch)

.. code-block:: python

   import numpy as np

   import drevalpy
   from drevalpy import ModelSpec
   from drevalpy.models.config import from_spec
   from drevalpy.plugin import FeatureFormat, FeatureFreePredictor, register_predictor


   @register_predictor(
       "meanBaseline",
       description="Predict the mean training response.",
       sample_input_format=FeatureFormat.NUMERIC_MATRIX,
       drug_input_format=FeatureFormat.NUMERIC_MATRIX,
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


   spec = ModelSpec("MeanBaseline", from_spec("meanBaseline"))
   results = drevalpy.run([spec], dataset, "LSO", hyperparameter_tuning=False)

``ModelSpec`` takes a name and a ``ModelConfig``. ``from_spec`` builds the config from a
recipe string; a feature-free predictor is just its own name. Passing a plain dict raises
``TypeError``.

Next steps
----------

- Test and package your component: :doc:`test_components`, :doc:`publish_plugin`.
- Make it visible to ``drevalpy`` commands: :doc:`load_extensions`.
- Recipe grammar and the hyperparameter key format: :doc:`/reference/recipes`,
  :doc:`/reference/hyperparameters`.
