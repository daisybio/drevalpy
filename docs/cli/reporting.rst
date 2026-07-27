Reporting
=========

After an experiment finishes, ``drevalpy report`` scores the predictions with
all available metrics and builds an HTML report with visualizations.

Options
-------

- ``--run_id`` — same run identifier you passed to ``drevalpy``
- ``--dataset_name`` — same dataset name used for the experiment
- ``--path_data`` — data directory (default ``data``)
- ``--result_path`` — results root (default ``results``)

Example
-------

.. code-block:: bash

   drevalpy report --run_id my_first_run --dataset_name TOYv1

The report is written under ``results/<run_id>/``. Open ``index.html`` in a
browser to inspect critical-difference diagrams, metric tables, and related
plots.

You need enough CV folds for the critical-difference diagram (typically at
least seven). See :doc:`experiment` for ``--n_cv_splits`` and
:doc:`/concepts/evaluation` for metrics.

Backward compatibility
----------------------

Legacy entry point
~~~~~~~~~~~~~~~~~~

Before 1.6.0, the report was commonly invoked as ``drevalpy-report``. Prefer
``drevalpy report``. The old console script remains a deprecated alias and may
be removed in a later version.
