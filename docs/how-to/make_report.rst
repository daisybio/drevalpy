Make a report
=============

.. interface: both

A report gathers the tables and plots of an experiment in one HTML file. This page shows how to
build it from saved results, normalize it against a reference model, choose its title and
location, and save single plots as images. :doc:`/tutorials/first_benchmark` shows how to read the
report.

The examples build the results of three models on the simulated dataset first. That is the
experiment the report describes.

Build a report from saved results
---------------------------------

``report`` reads an experiment directory, as written by ``drevalpy run`` or ``drevalpy aggregate``
(see :doc:`/reference/results`). Here ``NaiveMeanEffectsPredictor`` is the reference model; the
next section explains what that does. Leave ``--reference-model`` out to report the raw metrics.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         drevalpy data synthetic --output toy.h5mu
         drevalpy run NaiveDrugMeanPredictor NaiveMeanEffectsPredictor Ridge \
             --dataset toy.h5mu --split-mode LPO --no-hpo --output-dir results
         drevalpy report results --reference-model NaiveMeanEffectsPredictor --output-dir report

      The report command prints the progress of the plots and ends with:

      .. code-block:: text

         Report generated at report

   .. tab-item:: Python
      :sync: python

      ``create_report`` takes the result object itself, so there is no need to save it first.
      To report on results from disk, load them with ``ExperimentResult.load("results")``.

      .. code-block:: python

         import drevalpy
         from drevalpy import ModelSpec
         from drevalpy.visualization import create_report

         dataset = drevalpy.synthetic()
         names = ["NaiveDrugMeanPredictor", "NaiveMeanEffectsPredictor", "Ridge"]
         models = [ModelSpec.from_name(name) for name in names]
         results = drevalpy.run(models, dataset, "LPO", hyperparameter_tuning=False)
         create_report(results, "report", reference_model="NaiveMeanEffectsPredictor")

Open ``report/Drug-Response-Evaluation_multiqc_report.html`` in a browser. The report is a
single file, next to a ``_data`` directory with the tables behind it. The file name follows the
title, with spaces replaced by hyphens: a report called ``My title`` is
``My-title_multiqc_report.html``. Choose the title with ``--title`` (``-t``), or ``title=`` in
Python, and the directory with ``--output-dir`` (``-o``).

The report command reads only the predictions and metrics of the final models. It skips the
predictions of the tuning trials, which makes loading faster and uses less memory. It does not
read the dataset either, so ``report`` has no ``--dataset`` option.

Normalize against a reference model
-----------------------------------

A reference model removes what the easy part of the problem explains. With ``--reference-model``
(``reference_model=`` in Python), DrEvalPy subtracts the reference model's prediction for each
pair from both the observed response and the prediction of every other model, fold by fold,
and recomputes the metrics. Each model is then scored on what it adds beyond the reference.

The reference model is not in the normalized report; the other models are. Name a model that
is in the experiment, as it is named there. Otherwise the command stops with an error that lists
the models that are available. Without a reference model the report is not normalized, and
every metric is on the scale of the response. The experiment summary shows which case you have:
``Normalized by: None``.

``NaiveMeanEffectsPredictor`` is a good reference because it already predicts the mean plus the
effects of the sample and the drug; see :doc:`/how-to/choose_models`. Normalizing needs a
reference run for every fold of the other models, so run the reference model in the same
experiment.

Know which plots to expect
--------------------------

The report holds every plot that the results can support. Each plot lists the results it needs:
a plot that compares models needs several models, and a plot that shows the spread over folds
needs several folds. Plots whose requirements are not met are left out of the report; for
example, an experiment with a single model has no model comparison, critical difference diagram
or leaderboard. Some plots log a warning and skip a part of themselves, such as the Friedman test of the
critical difference diagram when only two models share the folds. ``drevalpy list visualizations`` shows the plots and their requirements;
see :doc:`/how-to/list_available`.

The log shows one line per plot with the time it took. ``drevalpy -v report ...`` also logs the
timing and memory of each model of a per-model plot, which is useful to find out which plot is slow
or runs out of memory.

Plots that you register with the visualization registry join the report when their requirements
are met; see :doc:`/how-to/extend/write_visualization`.

Save a plot as an image
-----------------------

For a figure in a paper or slides, save plots as PNG files. The plot classes are in
``drevalpy.visualization.plots``, one module per plot. A plot is computed from a result and
written with ``to_png``:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The command line writes the HTML report only. Use the Python tab to save images.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from drevalpy.visualization.plots.heatmap import HeatmapVisualization

         heatmap = HeatmapVisualization()
         heatmap.compute(results)
         heatmap.to_png("heatmap.png")

``save_all_png(results, "plots", reference_model=...)`` from ``drevalpy.visualization`` writes every
plot of the report as a PNG file into one directory, which takes about as long as building the
report.

Put the numbers in a table
--------------------------

The tables of the report are also available as data. ``results.summary_table()`` gives the pooled
metrics of each model and ``results.summary_table(group_by="drug")`` the metrics within each drug;
the command-line ``run`` and ``aggregate`` print the second with ``--group-by drug``. They are
plain dictionaries, ready for ``pandas.DataFrame``.
