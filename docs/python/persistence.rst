Persistence
===========

Trained models persist as a versioned joblib payload plus the public flat
hyperparameters applied at construction.

model.joblib
------------

``DRPModel.save`` writes a single native payload under the checkpoint directory:

- ``model.joblib`` — format name ``drevalpy-model``, resolved ``ModelConfig``,
  and fitted component state (``get_state`` / ``set_state`` on each component)

Public flat hyperparameters live on the instance after construction or
``load``; the checkpoint reconstructs them from the stored ``ModelConfig``.

.. code-block:: python

   from drevalpy.models import construct_model, load_model

   ElasticNet = construct_model("ElasticNet")
   model = ElasticNet()  # or ElasticNet({"alpha": 0.1})
   model.train(...)  # after a normal fit
   model.save("checkpoints/elastic_net")

   # When you already have the class handle:
   loaded = ElasticNet.load("checkpoints/elastic_net")

   # Or reconstruct entirely from the checkpoint (zoo or custom name):
   loaded = load_model("checkpoints/elastic_net")

Load only artifacts you created with the current native format in the same
drevalpy version family. Corrupted or unsupported payloads raise
``ModelCheckpointError`` subclasses (for example
``UnsupportedCheckpointFormatError`` or ``IncompatibleModelCheckpointError``).

Migration notes
---------------

Before 1.6.0, checkpoints stored pickled ``.model`` attributes,
standalone scalers, naive mean buffers, or the older ``composed_model.joblib``
layout. Those formats are **not** loadable. Retrain with the current release
and persist via ``model.save`` / ``ModelClass.load``.
