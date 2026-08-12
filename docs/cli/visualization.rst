Visualization and reporting
===========================

If you are reading this, we assume you are already familiar with this
concept:

- :doc:`/concepts/evaluation`

After an experiment finishes, ``drevalpy report`` builds an HTML report with
evaluation metrics and visualizations.

``drevalpy report``
-------------------

.. code-block:: bash

   drevalpy report EXPERIMENT_DIR [OPTIONS]

Arguments:

- ``EXPERIMENT_DIR`` — path to a saved ``ExperimentResult`` directory (the
  output of ``drevalpy run`` or ``drevalpy aggregate``).

Options:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``--output-dir`` / ``-o``
     - ``report``
     - Output directory for the HTML report.
   * - ``--title`` / ``-t``
     - ``Drug Response Evaluation``
     - Report title.
   * - ``--reference-model`` / ``-r``
     - None
     - Normalize metrics against this model.
   * - ``--dataset`` / ``-d``
     - None
     - Path to dataset ``.h5mu`` for metadata enrichment.

Example
-------

.. code-block:: bash

   drevalpy report results/ --output-dir report --title "My Benchmark"

With a reference model for normalized metrics:

.. code-block:: bash

   drevalpy report results/ \
       --output-dir report \
       --reference-model NaiveMeanEffectsPredictor \
       --dataset data/GDSC1.h5mu

The report uses `MultiQC <https://multiqc.info/>`_ internally and includes
critical-difference diagrams, metric tables, violin plots, heatmaps, and
scatter comparisons. You need enough CV folds (typically at least seven) for
meaningful critical-difference diagrams.

Evaluation concepts (normalized metrics, critical difference) are documented
in :doc:`/concepts/evaluation`. For the Python report API, see
:doc:`/python/visualization`.
