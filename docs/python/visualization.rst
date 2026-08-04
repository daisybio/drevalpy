Visualization and evaluation
============================

If you are reading this, we assume you are already familiar with this
concept:

- :doc:`/concepts/evaluation`

Score predictions with :func:`~drevalpy.evaluation.evaluate`, draw comparison
plots with the ``drevalpy.visualization`` classes, or render a full HTML
report with ``create_report``.

evaluate
--------

``evaluate`` scores a ``DrugResponseDataset`` that already has
``.predictions`` set:

.. code-block:: python

   from drevalpy.evaluation import evaluate

   metrics = evaluate(test_dataset, metric=["RMSE", "Pearson", "R^2"])
   # {"RMSE": ..., "Pearson": ..., "R^2": ...}

Available metric names include ``MSE``, ``RMSE``, ``MAE``, ``R^2``,
``Pearson``, ``spearman``, and ``kendall``.

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

Typical use after parsing result CSVs (as ``create_report`` does internally):

- ``Violin`` / ``Heatmap`` — metric distributions and heatmaps across models
- ``ComparisonScatter`` — model-vs-model scatter comparisons
- ``RegressionSliderPlot`` — true-vs-predicted regression views
- ``CriticalDifferencePlot`` — critical-difference diagrams over CV folds
- ``CrossStudyTables`` — transfer / cross-study summary tables

Most plot classes implement the ``OutPlot`` interface (``draw`` / write HTML
fragments). Prefer ``create_report`` unless you need a custom figure layout.

create_report
-------------

After ``drug_response_experiment`` finishes, build the HTML report from the
results directory:

.. code-block:: python

   from drevalpy.visualization.create_report import create_report

   create_report(
       run_id="my_first_run",
       dataset="TOYv1",
       path_data="data",
       result_path="results",
   )

This parses prediction CSVs under ``results/{run_id}``, writes aggregated
metric tables, and generates ``index.html`` plus per-test-mode pages you can
open in a browser. Evaluation concepts (normalized metrics, critical
difference) are documented in :doc:`/concepts/evaluation`. For the CLI report
command, see :doc:`/cli/reporting`.
