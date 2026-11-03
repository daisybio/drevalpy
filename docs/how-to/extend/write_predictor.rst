Write a predictor
=================

.. interface: python

A predictor is the model proper: it learns from a batch of sample-drug pairs and scores new
pairs. This page shows how to write one for each kind of input and how to declare its
hyperparameters and provenance. For the featurizers that feed it, see :doc:`write_featurizer`.

Declare what the predictor reads
--------------------------------

A predictor subclasses one of four bases from ``drevalpy.plugin`` and registers with
``register_predictor(name, *, description, tags=None, reference=None, accelerator=Accelerator.NONE,
license=None, artifact_license=None)``. What it reads from each
side is its ``sample_inputs`` and ``drug_inputs`` declaration, an ``Inputs``:

``Inputs.none()``
   No features on that side; config validation rejects a featurizer there.
``Inputs.matrix()``
   The side's numeric output, flattened into one design matrix.
``Inputs.blocks(*specs)``
   Named blocks, each a ``BlockSpec(name, format)``. A tuple of specs means that one of them
   must be emitted, and ``Inputs.blocks()`` without specs takes the featurizer's blocks as they
   are.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Base
     - Declares
     - You implement
   * - ``FeatureFreePredictor``
     - ``Inputs.none()`` on both sides
     - ``_fit(batch)``, ``_predict(batch)``
   * - ``MatrixPredictor``
     - ``Inputs.matrix()`` on both sides
     - ``_fit_matrix(x, y)``, ``_predict_matrix(x)``
   * - ``SingleDrugPredictor``
     - ``Inputs.none()`` on the drug side: set ``sample_inputs`` yourself
     - ``_fit_drug(drug_id, batch)``, ``_predict_drug(drug_id, model, batch)``
   * - ``Predictor``
     - Nothing: set ``sample_inputs`` and ``drug_inputs`` yourself
     - ``_fit(batch)``, ``_predict(batch)``

Set or override a declaration on a subclass, for example ``sample_inputs = Inputs.matrix()`` on
a ``SingleDrugPredictor`` (see :ref:`how-to/extend/write_predictor:Fit one model per drug`).
Registration rejects a predictor that declares neither. When a configuration is built, each side's declaration is checked against what its
featurizer emits, so a recipe that cannot work fails before training. A grid skips such a
combination and reports the reason; see :doc:`/how-to/evaluate_component_grids`.

The public ``fit`` and ``predict`` are implemented on the base class. ``fit`` raises
``ValueError`` for a batch without responses and drops pairs whose features are NaN;
``predict`` returns NaN for those pairs. Your methods never see them.

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

   .. tab-item:: Blocks

      A block predictor reads named blocks from ``batch.sample`` and ``batch.drug`` (each a
      ``FeaturizerResult``) instead of one flattened matrix. Blocks are indexed by entity, so
      expand them to one row per pair with ``batch.sample_pair_idx`` and
      ``batch.drug_pair_idx``, as the example does.

      .. literalinclude:: /examples/toy_block_predictor.py
         :language: python
         :caption: docs/examples/toy_block_predictor.py

Write a matrix predictor in a few lines
---------------------------------------

This predictor needs no scikit-learn. It solves a penalised least-squares problem on the design
matrix and declares one tunable hyperparameter.

.. code-block:: python

   import numpy as np

   from drevalpy.plugin import MatrixPredictor, register_predictor


   @register_predictor("leastSquares", description="Penalised least squares on the design matrix.")
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

Declare hyperparameters
-----------------------

Override the class method ``get_hyperparameter_space`` to declare the hyperparameters. Every
entry needs a ``default``: that value is used when nothing is tuned, and registration rejects a
space without one. A float entry also takes ``low``, ``high`` and ``log``, as in both examples
above. A setting that HPO must not draw is a ``{"type": "fixed", "default": value}`` entry, like
``fit_intercept`` of ``toyRidge``. The space is the only place defaults are declared:
``default_hyperparameters()`` reads them from it. The values in effect are available as
``self._hyperparameters``. The entry types are listed in :doc:`/reference/hyperparameters`.

Keep the fitted state in attributes
-----------------------------------

A saved model pickles the fitted predictor, so whatever ``_fit`` stores on ``self`` is saved and
restored with it; there is nothing to implement. ``is_fitted()`` is true once ``fit`` has
completed. An attribute that cannot be pickled, such as an open file or a training loop
object, has to be dropped in ``__getstate__``. :doc:`test_components` shows the check that a
predictor predicts the same after a pickle round trip.

.. warning::

   Saved models are serialized with joblib and pickle. Load only files you trust.

Credit the source of a ported model
-----------------------------------

``LiteratureReference`` is optional provenance metadata for components ported from a paper or
repository. Pass it as ``reference=`` to the register decorator, as ``toyRidge`` does above. It
does not change training, composition or saved models. It needs a ``repo_url`` (``http`` or
``https``), a citation (``citation_text``, ``citation_doi`` or both) and ``deviations``, which says
how the port differs from the original; constructing a reference without them raises
``ValueError``. A component with a reference also needs ``license=``, the license of the code it
was ported from; see the next section.

.. code-block:: python

   from drevalpy.plugin import LiteratureReference

   reference = LiteratureReference(
       repo_url="https://github.com/example/original-model",
       citation_text="Example et al., 2024.",
       deviations="Uses ridge regression instead of the original network.",
   )

Declare accelerator use and licenses
------------------------------------

Three more decorator arguments describe a component. ``drevalpy list predictors NAME`` and
``drevalpy describe MODEL`` show them, and none of them changes what the component computes.

``accelerator``
   An ``Accelerator`` from ``drevalpy.plugin``. ``Accelerator.NONE`` (the default) means the
   component runs on the CPU only. ``Accelerator.OPTIONAL`` means it uses a torch accelerator,
   such as a CUDA GPU or Apple MPS, when torch finds one, and the CPU otherwise.
   ``Accelerator.REQUIRED`` means it cannot run without one: ``DRPModel.train`` and ``predict``
   then raise ``DrEvalError`` on a machine without an accelerator. A model takes the strongest
   level of its predictor and featurizers. The level is only a declaration: pick the device
   yourself with torch, for example ``torch.accelerator.current_accelerator()``. The built-in
   components also honour ``DREVALPY_FORCE_CPU`` (:doc:`/reference/environment`).
``license``
   The SPDX license of the code the component was ported from, such as ``"MIT"`` or
   ``"GPL-3.0-only"``. It is required with ``reference=``; use ``"NOASSERTION"`` when that code
   has no license. Without it, the component has the license of the installed package that
   ships it, read from the package's ``License-Expression`` metadata (the ``license`` field of
   its ``pyproject.toml``). A component from a loose extension file has ``NOASSERTION``.
``artifact_license``
   The SPDX license of the third-party files the component downloads, such as pretrained
   weights, embeddings or gene sets. Leave it out when the component downloads none.

Registration rejects an expression that is not valid SPDX. ``NOASSERTION`` and ``LicenseRef-*``
ids are accepted.

.. code-block:: python

   import numpy as np

   from drevalpy.plugin import Accelerator, FeatureFreePredictor, register_predictor


   @register_predictor(
       "portedMean",
       description="Predict the mean training response.",
       reference=reference,
       accelerator=Accelerator.OPTIONAL,
       license="Apache-2.0",
       artifact_license="CC-BY-4.0",
   )
   class PortedMean(FeatureFreePredictor):
       def _fit(self, batch):
           self._mean = float(np.mean(batch.response))

       def _predict(self, batch):
           return np.full(batch.n_pairs, self._mean)

Fit one model per drug
----------------------

A single-drug predictor fits one model per drug and reads no drug features. Subclass
``SingleDrugPredictor``, declare ``sample_inputs`` and implement two hooks, each of which sees a
batch with the pairs of one drug:

``_fit_drug(drug_id, batch)``
   Returns the fitted model of the drug, or ``None`` to leave the drug without one.
``_predict_drug(drug_id, model, batch)``
   Scores the drug's pairs with what ``_fit_drug`` returned.

The base class routes the pairs by ``batch.drug_ids`` and predicts NaN for the pairs of a drug
it has no model for, such as a drug that was not in the training data. With ``--n-jobs``
(``n_jobs`` in Python) above 1 it fits the drugs in parallel worker processes, and it seeds
each drug from the run seed and the drug ID, so the fitted models do not depend on ``n_jobs``.
This example fits one ridge regression per drug:

.. literalinclude:: /examples/toy_single_drug_predictor.py
   :language: python
   :caption: docs/examples/toy_single_drug_predictor.py

``_fit_drug`` may run in a worker process, so make sure that it works there:

- **Do not start processes yourself.** Nested workers multiply the number of processes beyond
  ``n_jobs``. Use ``num_workers=0`` for a PyTorch ``DataLoader``, and leave the thread count of
  scikit-learn and PyTorch alone: the worker already limits it to one.
- **Do not write to a fixed path that the drugs share.** Write checkpoints below
  ``batch.training_context.checkpoint_dir``: in the batch of ``_fit_drug`` it is already a
  directory of the drug's own. Turn off the default logger of PyTorch Lightning, which
  writes to ``lightning_logs/`` in the working directory and makes concurrent fits race for the
  same version directory. The built-in neural predictors do it with
  ``default_logger=False`` in their ``LightningRun``; Weights & Biases logging stays on when a
  tracking run is active.
- **Return the fitted model instead of storing it on ``self``.** A worker fits a copy of the
  predictor, so an attribute that ``_fit_drug`` sets is lost. The base class collects the
  returned models in the parent predictor and saves them with it.

Read response layers
--------------------

A predictor can train on more of a pair than its response, such as the CurveCurator verdict
``regulation`` that ``SingleDrugMorgoth`` turns into classes. Name the response layers the fit
reads in the class attribute ``response_layers``. The fit then finds them in
``batch.response_layers``, a mapping from each name to one value per pair:

- **Training only.** The mapping is empty at prediction, so ``_predict`` must not read it.
- **Not transformed.** The response transformation changes ``batch.response`` but leaves the
  layers as the dataset stores them.
- **Possibly missing.** A layer the dataset lacks is left out of the mapping, so the fit must
  cope without it, for example with ``batch.response_layers.get(name)``.

This baseline predicts the mean response of the training pairs whose curve goes down, and the
mean of all pairs when the dataset has no ``regulation`` layer:

.. code-block:: python

   import numpy as np

   from drevalpy.plugin import FeatureFreePredictor, register_predictor


   @register_predictor("downMean", description="Predict the mean response of the down-regulated pairs.")
   class DownMean(FeatureFreePredictor):
       response_layers = ("regulation",)

       def _fit(self, batch):
           regulation = batch.response_layers.get("regulation")
           down = np.ones(batch.n_pairs, dtype=bool) if regulation is None else regulation == -1
           self._mean = float(np.mean(batch.response[down if down.any() else slice(None)]))

       def _predict(self, batch):
           return np.full(batch.n_pairs, self._mean)

``build_synthetic_batch(..., response_layers=)`` from ``drevalpy.testing`` builds a batch that
carries the layers, and the conformance checks below pass the predictor's ``response_layers``
for you.

Check that it works
-------------------

Run the shipped conformance checks on your predictor with a batch from ``drevalpy.testing``,
then use it in a model. The first block needs no dataset files; the second trains the
feature-free baseline defined below through the Python API.

.. code-block:: python

   from drevalpy import synthetic
   from drevalpy.testing import PREDICTOR_CHECKS, build_synthetic_batch

   dataset = synthetic(omics=["gene_expression"])
   batch = build_synthetic_batch(dataset, sample_block_names=("gene_expression",))
   for check in PREDICTOR_CHECKS:
       check(LeastSquaresPredictor, batch)

.. code-block:: python

   import numpy as np

   import drevalpy
   from drevalpy import ModelSpec
   from drevalpy.plugin import FeatureFreePredictor, register_predictor


   @register_predictor("meanBaseline", description="Predict the mean training response.")
   class MeanBaseline(FeatureFreePredictor):
       def _fit(self, batch):
           self._mean = float(np.mean(batch.response))

       def _predict(self, batch):
           return np.full(batch.n_pairs, self._mean)


   spec = ModelSpec.from_name("meanBaseline")
   results = drevalpy.run([spec], dataset, "LSO", hyperparameter_tuning=False)

``ModelSpec.from_name`` reads a name that is not a zoo preset as a recipe; a feature-free
predictor is a recipe of its own name. To report the model under another name, build the spec
as ``ModelSpec("MeanBaseline", from_recipe("meanBaseline"))`` with ``from_recipe`` from
``drevalpy.models.config``.

Next steps
----------

- Test and package your component: :doc:`test_components`, :doc:`publish_plugin`.
- Make it visible to ``drevalpy`` commands: :doc:`load_extensions`.
- Recipe grammar and the hyperparameter key format: :doc:`/reference/recipes`,
  :doc:`/reference/hyperparameters`.
