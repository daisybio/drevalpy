Quickstart
==========

Install DrEvalPy and its dependencies first — see
:doc:`/getting_started/installation`.

Load the GDSC1 dataset, resolve an ElasticNet specification, and run the
evaluation pipeline:

.. code-block:: python

   from drevalpy.data import load
   from drevalpy.models import ModelSpec
   from drevalpy import run, split

   dataset = load("GDSC1")

   elastic_net = ModelSpec.from_name("ElasticNet")

   result = run(
       models=[elastic_net],
       dataset=dataset,
       split_mode="LSO",
       hyperparameter_tuning=False,
   )

The experiment runner constructs a fresh ``DRPModel`` from each specification
for every fold. For a single-model script outside the experiment runner:

.. code-block:: python

   from drevalpy.models import DRPModel

   model = DRPModel(elastic_net, hyperparameters={"alpha": 0.1})
   fold = split(dataset, mode="LSO", n_splits=2)[0]
   model.train(dataset, fold.train)
   prediction = model.predict(dataset, fold.test)

:func:`~drevalpy.run` returns an
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
