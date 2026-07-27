Persistence
===========

Trained native models persist as a versioned joblib payload plus the public
flat hyperparameters applied at construction.

composed_model.joblib
---------------------

``ComposedModel.save`` / ``NativeDRPModel.save`` write a single native payload:

- ``composed_model.joblib`` — format name/version, resolved ``ModelConfig``,
  and fitted component state (``get_state`` / ``set_state`` on each component)

Public flat hyperparameters live on the facade instance after construction or
``load``; the checkpoint reconstructs them from the stored ``ModelConfig``.

.. code-block:: python

   from drevalpy.models import construct_model

   ElasticNet = construct_model("ElasticNet")
   model = ElasticNet()  # or ElasticNet({"alpha": 0.1})
   model.train(...)  # after a normal fit
   model.save("checkpoints/elastic_net")

   loaded = ElasticNet.load("checkpoints/elastic_net")

You can also save/load the underlying stack without the DRPModel facade:

.. code-block:: python

   from drevalpy.models.config import ModelConfig
   from drevalpy.models.composed_model import ComposedModel

   composed = ModelConfig.from_spec("ElasticNet").create_model()
   # ... fit composed ...
   composed.save("checkpoints/elastic_net")
   restored = ComposedModel.load("checkpoints/elastic_net")

Load only artifacts you created with the current native format in the same
drevalpy version family. Corrupted or unsupported payloads raise
``ComposedModelCheckpointError`` subclasses.

Backward compatibility
----------------------

No longer supported
~~~~~~~~~~~~~~~~~~~

Before 1.6.0, checkpoints stored pickled ``.model`` attributes,
standalone scalers, or naive mean buffers. Those formats are **not** loadable.
Retrain with the current release and persist via ``composed_model.joblib``.
