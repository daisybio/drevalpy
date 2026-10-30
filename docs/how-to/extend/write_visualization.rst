Write a visualization
=====================

.. interface: python

A visualization is a plot of experiment or model results that the report generator can pick
up automatically. This page shows how to write a plot, register it, and control when the
report includes it.

Subclass ``Plot``
-----------------

A plot subclasses ``Plot`` from ``drevalpy.plugin`` and is stateless: every method takes the
result it works on, so one instance can draw any number of results.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Responsibility
   * - ``figure(result)``
     - Draw the result and return a Matplotlib or Plotly figure. You implement this.
   * - ``sections(result)``
     - Return the ``Section`` objects of the report. The default embeds ``figure(result)``,
       which must then be a Matplotlib figure, as one PNG section named after the plot.
       Override it for a Plotly plot or for several sections.
   * - ``result_type``
     - ``ExperimentResult`` (the default) for a plot across all models, or ``ModelResult`` for
       a plot of one model. The report draws a ``ModelResult`` plot once per model.
   * - ``requirements``
     - A frozenset of ``PlotRequirement`` values, empty by default. The report only includes
       the plot for experiments that satisfy all of them.

``save_png(plot, result, path)`` writes the figure of a plot to a PNG file, and
``show(plot, result)`` displays it in a notebook.

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

.. literalinclude:: /examples/toy_visualization.py
   :language: python
   :caption: docs/examples/toy_visualization.py

Register the plot
-----------------

``register_visualization(name, description="", *, replace=False)`` registers the class and sets
its ``name``. A class that is not a ``Plot`` raises ``TypeError``, and a name that is already
registered raises ``ValueError`` unless you pass ``replace=True``.

Try it
------

This plot draws predicted against observed response for one model. It runs on a hand-built
``ModelResult``, so you can try it without an experiment on disk.

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   from drevalpy.plugin import ModelResult, Plot, register_visualization, save_png
   from drevalpy.types.results import RunResult


   @register_visualization("predictionScatter", "Predicted against observed response for one model.")
   class PredictionScatter(Plot):
       """Scatter the pooled predictions of every fold."""

       result_type = ModelResult

       def figure(self, result):
           observed = np.concatenate([run.ground_truth for run in result.runs])
           predicted = np.concatenate([run.predictions for run in result.runs])
           figure, axes = plt.subplots(figsize=(4, 4))
           axes.scatter(observed, predicted, s=4)
           axes.set_xlabel("observed")
           axes.set_ylabel("predicted")
           axes.set_title(result.model_name)
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
   result = ModelResult(model_name="demo", dataset_name="SYNTHETIC", runs=[run])
   plot = PredictionScatter()
   save_png(plot, result, "scatter.png")
   assert plot.sections(result)

Use the plot in a report
------------------------

``applicable(experiment)`` from ``drevalpy.registry.visualization`` returns the registered
classes whose requirements an experiment satisfies. The report generator
(``drevalpy.visualization.report.create_report``, which needs the ``multiqc`` extra) runs every
applicable plot. The plot has to be registered in the process that builds the report, so to see
yours from the command line, load it with ``-e`` before the subcommand, as in
``drevalpy -e my_extensions report ...``. :doc:`load_extensions` explains the flag, and
:doc:`/how-to/make_report` covers the report itself.

There is no conformance check for plots. Exercise yours as above: call ``sections`` and
``save_png`` on a hand-built result.
