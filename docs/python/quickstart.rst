Quickstart
==========

Install DrEvalPy and its dependencies first — see
:doc:`/getting_started/installation`.

Load the TOYv1 response table, resolve ElasticNet from the model zoo, and hand
the model class to :func:`~drevalpy.experiment.drug_response_experiment`:

.. code-block:: python

   from drevalpy.datasets.loader import load_dataset
   from drevalpy.experiment import drug_response_experiment
   from drevalpy.models import construct_model

   response_data = load_dataset("TOYv1", path_data="data")

   ElasticNet = construct_model("ElasticNet")

   drug_response_experiment(
       models=[ElasticNet],
       response_data=response_data,
       run_id="my_first_run",
       test_mode="LCO",
       path_data="data",
       path_out="results/",
       hyperparameter_tuning=False,
   )

:func:`~drevalpy.models.construct_model` returns a **class**. The experiment
call expects the class (or a list of classes) so each CV fold can construct a
fresh configured instance (``Model()`` or ``Model(best_hpams)``) and call
``train``. For a single-model script outside the experiment runner:

.. code-block:: python

   model = ElasticNet()  # or ElasticNet({"alpha": 0.1})
   model.train(...)

Results land under ``results/my_first_run/TOYv1/LCO``. See
:doc:`visualization` for ``create_report``, :doc:`datasets` for other screens
and custom tables, and :doc:`experiments` for baselines, tuning, and
stress-test options.

After concepts
--------------

Shared vocabulary lives under **Concepts**. Use this map when you leave the
concepts track for Python:

- :doc:`/concepts/datasets` → :doc:`datasets`
- :doc:`/concepts/evaluation` → :doc:`experiments` and :doc:`visualization`
- :doc:`/concepts/component_catalog` → :doc:`custom_models`
- :doc:`/concepts/from_components_to_models` → :doc:`architecture` and
  :doc:`hyperparameter_tuning`
- :doc:`/concepts/model_zoo` → :doc:`models`

Migration note: ``MODEL_FACTORY``
---------------------------------

Before 1.6.0, models were resolved through ``MODEL_FACTORY`` (and the
multi-/single-drug variants). Those dicts remain lazy **built-in-only**
compatibility views equal to ``construct_model(name)`` for zoo preset names,
but emit ``FutureWarning`` and may be removed in a future release. Prefer
``construct_model``:

.. code-block:: python

   # Preferred
   from drevalpy.models import construct_model

   ElasticNet = construct_model("ElasticNet")

   # Equivalent for built-in zoo names (deprecated)
   from drevalpy.models import MODEL_FACTORY

   ElasticNet = MODEL_FACTORY["ElasticNet"]

The old ``MODEL_FACTORY`` cannot resolve custom recipe strings or externally
registered zoo entries. ``construct_model`` covers those paths. Further
migration notes live on :doc:`models`, :doc:`architecture`, and
:doc:`persistence`.
