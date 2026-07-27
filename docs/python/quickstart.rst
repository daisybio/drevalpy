Quickstart
==========

Install DrEvalPy and its dependencies first — see
:doc:`/getting_started/installation`.

Load the TOYv1 response table, resolve ElasticNet from the model zoo, build it
with default hyperparameters, and hand the facade class to
``drug_response_experiment``:

.. code-block:: python

   from drevalpy.datasets.loader import load_dataset
   from drevalpy.experiment import drug_response_experiment
   from drevalpy.models import construct_model

   response_data = load_dataset("TOYv1", path_data="data")

   ElasticNet = construct_model("ElasticNet")
   model = ElasticNet()
   model.build_model(model.get_default_hyperparameters())

   drug_response_experiment(
       models=[ElasticNet],
       response_data=response_data,
       run_id="my_first_run",
       test_mode="LCO",
       path_data="data",
       path_out="results/",
       hyperparameter_tuning=False,
   )

``construct_model`` returns a **class**. Instantiating it and calling
``build_model`` is enough for a single-model workflow; the experiment call
expects the class (or a list of classes) so each CV fold can construct a fresh
instance.

Results land under ``results/my_first_run/TOYv1/LCO``. See
:doc:`visualization` for ``create_report``, :doc:`datasets` for other screens
and custom tables, and :doc:`experiments` for tuning and stress-test options.

Backward compatibility
----------------------

MODEL_FACTORY
~~~~~~~~~~~~~

Before 1.6.0, models were resolved through ``MODEL_FACTORY`` (and the
multi-/single-drug variants). This remains available for backward
compatibility, but is deprecated and may be removed in a future release.
Prefer ``construct_model``:

.. code-block:: python

   from drevalpy.models import construct_model

   ElasticNet = construct_model("ElasticNet")
