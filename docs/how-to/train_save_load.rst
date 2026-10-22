Train, save and load a model
============================

.. interface: python

An experiment trains a model on every fold and keeps its predictions, not the model. This page
shows how to train one model yourself, score it, save the fitted model to a file, and load it in
another session. The command line has no command for this: ``drevalpy run`` and ``drevalpy
single`` write predictions, not model files.

The examples train ``ElasticNet`` on one fold of the simulated dataset.

Train a model on a fold
-----------------------

A ``DRPModel`` is a model that you train and predict with. Build it from a ``ModelSpec`` (see
:doc:`/how-to/choose_models`) and the hyperparameters you want. The values below are examples; the
model fits with exactly these values and does not tune. :doc:`/how-to/tune_hyperparameters` shows how to
search for them.

``train`` and ``predict`` take a **scope**: a ``SplitMask`` that says which pairs to use. A fold from
:doc:`/how-to/load_and_split` has the scopes ``train``, ``val`` and ``test``. Train on
``train_val``, the training and validation pairs together, as ``run`` does for the final fit,
and predict the ``test`` pairs.

.. code-block:: python

   import drevalpy
   from drevalpy import DRPModel, ModelSpec

   dataset = drevalpy.synthetic()
   fold = drevalpy.split(dataset, "LPO")[0]

   spec = ModelSpec.from_name("ElasticNet")
   model = DRPModel(spec, hyperparameters={"alpha": 0.1, "l1_ratio": 0.2})
   model.train(dataset, fold.train_val)
   predictions = model.predict(dataset, fold.test)
   print(predictions.shape)

.. code-block:: text

   (92,)

``predict`` returns one value for each pair of the scope, in the order of ``scope.pairs``, in the
units of the response. A hyperparameter is a short name such as ``alpha``, or a qualified key
when several components share a name; see :doc:`/reference/hyperparameters`. A name that no
component has raises ``ValueError``.

Predictors that stop training early, such as the neural networks, also need an
``early_stopping_scope`` that does not overlap the training scope. ``spec.requires_early_stopping``
says whether a model does, and ``fold.early_stopping_mask()`` splits one off the validation
pairs.

Score the predictions
---------------------

The observed responses for the same pairs are in the response matrix, which is indexed by the
sample and drug of each pair. Pass both to ``evaluate``:

.. code-block:: python

   from drevalpy.evaluation import evaluate

   truth = dataset.response_matrix[tuple(fold.test.pairs.T)]
   scores = evaluate(predictions, truth, metric=["Pearson", "RMSE"])
   print({name: round(value, 3) for name, value in scores.items()})

.. code-block:: text

   {'Pearson': 0.891, 'RMSE': 0.782}

``evaluate`` scores all the pairs together. To score within each drug or sample, see
:doc:`/reference/metrics`.

Scale the response
------------------

Pass an unfitted scikit-learn transformer as ``response_transformation`` to train on a scaled
response. The model fits it on the training pairs and transforms the predictions back, so
``predict`` still returns values in the units of the response:

.. code-block:: python

   from sklearn.preprocessing import StandardScaler

   scaled_model = DRPModel(spec, {"alpha": 0.1, "l1_ratio": 0.2}, response_transformation=StandardScaler())
   scaled_model.train(dataset, fold.train_val)

Save a model
------------

``save`` writes the fitted model, with everything it needs to predict, to one ZIP archive. DrEvalPy
appends ``.zip`` to the path if it is missing. Saving a model that is not trained raises
``RuntimeError``.

.. code-block:: python

   model.save("elastic_net")

The call writes ``elastic_net.zip``. The path may be remote, such as an ``s3://`` URL, if the
matching ``fsspec`` backend is installed.

Load a model
------------

``DRPModel.load`` restores the model, including its response transformation, and predicts the
same values:

.. code-block:: python

   import numpy as np

   loaded = DRPModel.load("elastic_net.zip")
   print(np.allclose(loaded.predict(dataset, fold.test), predictions))

.. code-block:: text

   True

Three things to know before you share or keep a checkpoint:

**Load only files you trust.** A checkpoint is a pickle. Loading one can run arbitrary code, so
load only checkpoints you made yourself or got from a source you trust.

**Register your own components first.** The checkpoint stores the names of the model's
components and their fitted state, not their code. If the model uses a featurizer or a predictor
that you wrote, register it before you call ``load``, for example by loading your extension
directory; see :doc:`/how-to/extend/load_extensions`.

**Checkpoints do not cross format versions.** The current format version is 4. A checkpoint from
an older format is refused with a message to retrain the model and save it again. Keep checkpoints
for the DrEvalPy version that wrote them.

Control where training checkpoints go
-------------------------------------

Components can write checkpoints while they train. ``model_checkpoint_dir`` sets the directory
for them. It is not the file that
``save`` writes. With ``None``, the default, DrEvalPy uses a temporary directory and removes it
when training ends:

.. code-block:: python

   model.train(dataset, fold.train_val, model_checkpoint_dir="training_checkpoints")

Training a model again replaces its fitted state.
