Write a visualization
=====================

.. interface: python

A visualization is a plot of experiment or model results that the report generator can pick
up automatically. This page shows how to subclass the Matplotlib helper, register the plot,
and control when the report includes it.

The four methods of a visualization
-----------------------------------

``Visualization`` declares four abstract methods:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Method
     - Responsibility
   * - ``compute(result, dataset=None)``
     - Derive the plot data from the result and store it on the instance.
   * - ``to_png(path)``
     - Write a static image.
   * - ``to_multiqc()``
     - Return ``Section`` objects for the report.
   * - ``show()``
     - Display the plot interactively in a notebook.

For a static Matplotlib plot, subclass ``ImageVisualization`` from ``drevalpy.plugin`` instead.
It implements ``to_png``, ``to_multiqc`` and ``show`` in terms of a figure, so you supply only
``compute``, which must assign the figure to ``self._fig``, and ``_create_figure``. Until
``compute`` has run, the three renderers raise ``RuntimeError``.

.. literalinclude:: /examples/toy_visualization.py
   :language: python
   :caption: docs/examples/toy_visualization.py

Choose what the plot receives
-----------------------------

``register_visualization(name, description="", *, result_type="ExperimentResult",
requirements=frozenset(), override=False)`` takes these arguments:

- ``result_type`` is ``"ExperimentResult"`` for a plot across all models, or ``"ModelResult"``
  for a plot of one model. The report runs a ``ModelResult`` plot once per model.
- ``requirements`` is a frozenset of ``PlotRequirement`` values. The report only includes the
  plot for experiments that satisfy all of them.

  .. list-table::
     :header-rows: 1
     :widths: 30 70

     * - Requirement
       - The experiment must have
     * - ``MULTIPLE_MODELS``
       - More than one model.
     * - ``MULTIPLE_FOLDS``
       - More than one cross-validation fold.
     * - ``RANDOMIZATION``
       - Randomization test results.
     * - ``ROBUSTNESS``
       - Robustness test results.

- A name that is already registered raises ``ValueError`` unless you pass ``override=True``.

Try it
------

This plot draws predicted against observed response for one model. It runs on a hand-built
``ModelResult``, so you can try it without an experiment on disk.

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   from drevalpy.plugin import ImageVisualization, ModelResult, RunResult, register_visualization


   @register_visualization(
       "predictionScatter",
       "Predicted against observed response for one model.",
       result_type="ModelResult",
   )
   class PredictionScatter(ImageVisualization):
       """Scatter the pooled predictions of every fold."""

       def __init__(self):
           self._pairs = None
           self._title = ""

       def compute(self, result, dataset=None):
           observed = np.concatenate([run.ground_truth for run in result.runs])
           predicted = np.concatenate([run.predictions for run in result.runs])
           self._pairs = (observed, predicted)
           self._title = result.model_name
           self._fig = self._create_figure()

       def _create_figure(self):
           figure, axes = plt.subplots(figsize=(4, 4))
           axes.scatter(*self._pairs, s=4)
           axes.set_xlabel("observed")
           axes.set_ylabel("predicted")
           axes.set_title(self._title)
           return figure


   rng = np.random.default_rng(0)
   run = RunResult(
       model_name="demo",
       dataset_name="SYNTHETIC",
       fold_index=0,
       predictions=rng.normal(size=32),
       ground_truth=rng.normal(size=32),
       sample_ids=np.array([f"S{index:03d}" for index in range(32)]),
       drug_ids=np.array(["1"] * 32),
   )
   plot = PredictionScatter()
   plot.compute(ModelResult(model_name="demo", dataset_name="SYNTHETIC", runs=[run]))
   plot.to_png("scatter.png")
   assert plot.to_multiqc()

Use the plot in a report
------------------------

``visualization_registry.applicable(experiment)`` from ``drevalpy.registry.visualization``
returns the registered classes whose requirements an experiment satisfies. The report generator
(``drevalpy.visualization.report.create_report``, which needs the ``multiqc`` extra) runs every
applicable plot. The plot has to be registered in the process that builds the report, so to see
yours from the command line, load it with ``-e`` before the subcommand, as in
``drevalpy -e my_extensions report ...``. :doc:`load_extensions` explains the flag, and
:doc:`/how-to/make_report` covers the report itself.

There is no conformance check for visualizations yet. Exercise yours as above: call
``compute`` first, then the renderers.
