Quickstart
==========

Install DrEvalPy and its dependencies first — see
:doc:`/getting_started/installation`.

Load the GDSC1 dataset, resolve ElasticNet from the model zoo, and run the
evaluation pipeline:

.. code-block:: python

   from drevalpy.data import load
   from drevalpy.models import construct_model
   from drevalpy.run import run

   dataset = load("GDSC1")

   ElasticNet = construct_model("ElasticNet")

   result = run(
       models=[ElasticNet],
       dataset=dataset,
       split_mode="LCO",
       hyperparameter_tuning=False,
   )

:func:`~drevalpy.models.construct_model` returns a **class**. The pipeline
expects classes (or a list of classes) so each CV fold can construct a
fresh configured instance and call ``train``. For a single-model script
outside the experiment runner:

.. code-block:: python

   model = ElasticNet()  # or ElasticNet({"alpha": 0.1})
   model.train(...)

:func:`~drevalpy.run.run` returns an
:class:`~drevalpy.types.results.ExperimentResult` that groups predictions,
metrics, and metadata for all folds. Save it and generate a report:

.. code-block:: python

   result.save("results/")

   from drevalpy.visualization.report import create_report

   create_report(result, "report/")

See :doc:`visualization` for report options, :doc:`datasets` for loading and
splitting, and :doc:`experiments` for tuning and stress-test options.

After concepts
--------------

Shared vocabulary lives under **Concepts**. Use this map when you leave the
concepts track for Python:

- :doc:`/concepts/datasets` → :doc:`datasets`
- :doc:`/concepts/evaluation` → :doc:`experiments` and :doc:`visualization`
- :doc:`/concepts/component_catalog` → :doc:`extensions`
- :doc:`/concepts/from_components_to_models` → :doc:`models` and
  :doc:`experiments`
- :doc:`/concepts/model_zoo` → :doc:`models`
- :doc:`/concepts/registries` → :doc:`extensions`
