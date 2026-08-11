Visualization and evaluation
============================

If you are reading this, we assume you are already familiar with this
concept:

- :doc:`/concepts/evaluation`

Score predictions with :func:`~drevalpy.evaluation.evaluate`, draw comparison
plots with the ``drevalpy.visualization`` classes, or render a full HTML
report with :func:`~drevalpy.visualization.report.create_report`.

evaluate
--------

``evaluate`` computes one or more metrics given predictions and observed
response values:

.. code-block:: python

   import numpy as np
   from drevalpy.evaluation import evaluate

   predictions = np.array([1.2, 3.4, 2.5])
   response = np.array([1.0, 3.5, 2.0])

   metrics = evaluate(predictions, response, metric=["RMSE", "Pearson", "R^2"])
   # {"RMSE": ..., "Pearson": ..., "R^2": ...}

You can also pass an object that has ``.predictions`` and ``.response``
attributes:

.. code-block:: python

   metrics = evaluate(run_result, metric="Pearson")

Available metric names include ``MSE``, ``RMSE``, ``MAE``, ``R^2``,
``Pearson``, ``Spearman``, and ``Kendall``.

Plot classes
------------

The visualization package exports plot helpers used by the report pipeline:

.. code-block:: python

   from drevalpy.visualization import (
       ComparisonScatter,
       CriticalDifferencePlot,
       CrossStudyTables,
       Heatmap,
       RegressionSliderPlot,
       Violin,
   )

Typical use after an experiment (as ``create_report`` does internally):

- ``Violin`` / ``Heatmap`` — metric distributions and heatmaps across models
- ``ComparisonScatter`` — model-vs-model scatter comparisons
- ``RegressionSliderPlot`` — true-vs-predicted regression views
- ``CriticalDifferencePlot`` — critical-difference diagrams over CV folds
- ``CrossStudyTables`` — transfer / cross-study summary tables

Most plot classes implement the ``Visualization`` interface (``compute`` and
``to_multiqc``). Prefer ``create_report`` unless you need a custom figure
layout.

create_report
-------------

After :func:`~drevalpy.run.run` finishes, build the HTML report from the
result object:

.. code-block:: python

   from drevalpy.visualization.report import create_report

   create_report(result, "report/")

Or load a previously saved experiment:

.. code-block:: python

   from drevalpy.types.results import ExperimentResult
   from drevalpy.visualization.report import create_report

   experiment = ExperimentResult.load("results/")
   create_report(experiment, "report/", title="My Benchmark")

Parameters:

- ``result`` — an ``ExperimentResult``, ``ModelResult``, or ``RunResult``
- ``output_dir`` — where to write the HTML report
- ``title`` — report title (default ``"Drug Response Evaluation"``)
- ``reference_model`` — if set, normalize metrics against this model
- ``dataset`` — optional ``Dataset`` for drug/cell-line metadata in plots

The report uses `MultiQC <https://multiqc.info/>`_ internally. Evaluation
concepts (normalized metrics, critical difference) are documented in
:doc:`/concepts/evaluation`. For the CLI report command, see
:doc:`/cli/visualization`.
