Train, save and load a model
============================

.. interface: both

An experiment trains a model on every fold and keeps its predictions, not the model. This page
shows how to train a **final model** on all the pairs of a dataset, save it, and apply it to new
data, on the command line and in Python. The Python sections then show how to train a model on one
fold yourself, score it, and scale its response.

The examples train ``ElasticNet`` on the simulated dataset.

Train a final model on all pairs
--------------------------------

A final model is trained on every usable pair of a dataset, with no test set, for use on data you
have not seen yet. Like a run, it sets aside a random 10% of the pairs as a validation slice, and
it applies the curve-quality rule to choose the usable pairs. A model that stops early keeps a
quarter of the slice as its early-stopping set and trains on every other pair; the other models
train on all of them. A single-drug model fits one estimator per drug, as it does in a run.

Without ``--no-hpo`` (``hyperparameter_tuning=False`` in Python) the model first tunes its
hyperparameters on the validation slice. The fitted response transformation is part of the saved
model, so the predictions are in the units of the response.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      ``drevalpy train MODEL --dataset DATASET --output MODEL_DIR`` trains the model and writes the
      directory ``MODEL_DIR``. It holds ``model.zip``, the checkpoint that ``DRPModel.save`` writes,
      and ``metadata.json``, which records the model, the dataset, the seed, the response
      transformation and the version of DrEvalPy. ``drevalpy predict MODEL_DIR --dataset DATASET
      --output PREDICTIONS`` loads the checkpoint and predicts every pair of the dataset that has an
      observed response.

      .. code-block:: bash

         drevalpy data synthetic --output toy.h5mu
         drevalpy train ElasticNet --dataset toy.h5mu --output final_model --no-hpo
         drevalpy predict final_model --dataset toy.h5mu --output predictions.npz

      ``train`` takes the options ``--hpo-metric``, ``--hpo-trials``, ``--hpo-random-state`` and
      ``--seed`` of ``run``, ``--response-transformation``, and the curve-quality options
      ``--no-quality-filter``, ``--min-relevance-score`` and ``--min-abs-fold-change``. The dataset
      of ``predict`` can be another dataset than the one the model was trained on; it needs the
      views the model reads. ``predictions.npz`` has the layout of a run file; see
      :ref:`predictions-file`.

   .. tab-item:: Python
      :sync: python

      ``drevalpy.train`` takes a model (a ``ModelSpec`` or a zoo name) and a dataset (an object
      or a name) and returns the fitted ``DRPModel``:

      .. code-block:: python

         import numpy as np

         import drevalpy
         from drevalpy import DRPModel
         from drevalpy.types import SplitMask

         dataset = drevalpy.synthetic()
         final_model = drevalpy.train("ElasticNet", dataset, hyperparameter_tuning=False)
         final_model.save("final_elastic_net")

         every_pair = SplitMask(np.isfinite(dataset.response_matrix))
         applied = DRPModel.load("final_elastic_net.zip").predict(dataset, every_pair)
         print(applied.shape)

      .. code-block:: text

         (456,)

      ``train`` also takes ``response_transformation``, ``quality`` (a ``CurveQuality``), ``seed``
      and the tuning options ``hpo_metric``, ``hpo_num_trials`` and ``hpo_random_state``. The
      sections below explain ``save`` and ``load``.

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

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command line has no command that trains on a fold and keeps the model.
      ``drevalpy single`` trains on one fold, but writes its predictions and metrics, not the
      model. Use ``drevalpy train`` above for a model you keep.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy import ModelSpec

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

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      ``drevalpy single`` and ``drevalpy run`` score the test fold themselves. ``drevalpy predict``
      scores the predictions it writes; the scores are in the ``metrics`` of its output file.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy.evaluation import evaluate

         truth = dataset.response_matrix[tuple(fold.test.pairs.T)]
         scores = evaluate(predictions, truth, ["Pearson", "RMSE"])
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

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      ``drevalpy train`` takes ``--response-transformation`` with ``none`` (default),
      ``standard``, ``minmax`` or ``robust``. The command fits the transformation on the
      training pairs and ``drevalpy predict`` transforms the predictions back.

      .. code-block:: bash

         drevalpy train ElasticNet --dataset toy.h5mu --output scaled_model --no-hpo --response-transformation standard

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from sklearn.preprocessing import StandardScaler

         scaled_model = DRPModel(spec, {"alpha": 0.1, "l1_ratio": 0.2}, response_transformation=StandardScaler())
         scaled_model.train(dataset, fold.train_val)

Save a model
------------

``save`` writes the fitted model, with everything it needs to predict, to one ZIP archive. DrEvalPy
appends ``.zip`` to the path if it is missing. Saving a model that is not trained raises
``RuntimeError``.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      ``drevalpy train`` saves the model it trains as ``model.zip`` in its output directory.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         model.save("elastic_net")

      The call writes ``elastic_net.zip``. The path may be remote, such as an ``s3://`` URL, if the
      matching ``fsspec`` backend is installed.

Load a model
------------

``DRPModel.load`` restores the model, including its response transformation, and predicts the
same values:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      ``drevalpy predict`` loads the checkpoint in a model directory. It also takes the path of a
      checkpoint file, such as one written by ``DRPModel.save``. A path that does not exist, or a
      directory without ``model.zip``, is reported as an ``Error:`` line.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         loaded = DRPModel.load("elastic_net.zip")
         print(np.allclose(loaded.predict(dataset, fold.test), predictions))

      .. code-block:: text

         True

Three things to know before you share or keep a checkpoint:

**Load only files you trust.** A checkpoint is a pickle. Loading one can run arbitrary code, so
load only checkpoints you made yourself or got from a source you trust. This applies to
``drevalpy predict`` too.

**Register your own components first.** The checkpoint stores the names of the model's
components and their fitted state, not their code. If the model uses a featurizer or a predictor
that you wrote, register it before you call ``load``, for example by loading your extension
directory; see :doc:`/how-to/extend/load_extensions`. On the command line, give the directory with
``drevalpy --extensions-dir DIR predict ...``.

**Checkpoints do not cross format versions.** A checkpoint records the format version that wrote it,
and ``load`` refuses a checkpoint whose version differs from the one your DrEvalPy reads, with an
``UnsupportedCheckpointFormatError`` that asks you to retrain the model and save it again. Keep
checkpoints for the DrEvalPy version that wrote them.

Control where training checkpoints go
-------------------------------------

Components can write checkpoints while they train. ``model_checkpoint_dir`` sets the directory
for them. It is not the file that
``save`` writes. With ``None``, the default, DrEvalPy uses a temporary directory and removes it
when training ends:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command line always uses a temporary directory.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         model.train(dataset, fold.train_val, model_checkpoint_dir="training_checkpoints")

Training a model again replaces its fitted state.
