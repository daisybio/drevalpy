Metrics and response transformations
====================================

:doc:`/explanation/metrics` explains why pooled scores can mislead and why the per-drug
metrics and a naive baseline matter. This page lists the metrics and how they are computed.

Metrics
-------

``drevalpy.evaluation.AVAILABLE_METRICS`` maps each name to a function. The names are
case-sensitive and ``R^2`` is written with a caret.

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Name
     - Better when
     - Computed with
   * - ``MSE``
     - lower
     - ``sklearn.metrics.mean_squared_error``
   * - ``RMSE``
     - lower
     - ``sklearn.metrics.root_mean_squared_error``
   * - ``MAE``
     - lower
     - ``sklearn.metrics.mean_absolute_error``
   * - ``R^2``
     - higher
     - ``sklearn.metrics.r2_score``
   * - ``Pearson``
     - higher
     - ``scipy.stats.pearsonr``
   * - ``Spearman``
     - higher
     - ``scipy.stats.spearmanr``
   * - ``Kendall``
     - higher
     - ``scipy.stats.kendalltau``

``MINIMIZATION_METRICS`` holds the three error metrics, ``MAXIMIZATION_METRICS`` the other
four, and ``get_mode(metric)`` returns ``"min"`` or ``"max"``.

Entry points
~~~~~~~~~~~~

``evaluate(predictions, response, metric="Pearson")``
   Scores one metric name or a list of names.

``score_predictions(predictions, response, metrics=None, *, empty_as_dict=False)``
   Scores every metric if ``metrics`` is ``None``. With ``empty_as_dict=True`` it returns
   ``{}`` when no pair is usable.

Both take one-dimensional arrays of equal length and raise ``ValueError`` for an unknown
metric.

Missing and degenerate input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pair is dropped if the prediction or the response is NaN. Then:

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - Situation
     - Score
   * - Fewer than one pair, for ``MSE``, ``RMSE`` or ``MAE``
     - ``NaN``
   * - Fewer than two pairs, for ``R^2`` or a correlation
     - ``NaN``
   * - Constant predictions (within ``1e-6``), for a correlation
     - ``0.0``
   * - Constant response (within ``1e-6``), for a correlation
     - ``NaN``

Constant predictions are tested first, so a correlation with both constant is ``0.0``.

Pooled and per-group scores
---------------------------

By default, a fold's metric is pooled: it is computed over all of the fold's pairs, and the
result is the mean over folds.

``ExperimentResult.summary_table(*, group_by=None)`` returns ``{model: {metric: mean}}``.
With ``group_by="drug"`` or ``group_by="sample"`` each metric is computed within every drug
or sample of a fold, averaged over the groups where it is defined, and then averaged over
folds. Any other ``group_by`` value raises ``ValueError``. ``summary_group_counts(group_by)``
returns the mean number of groups behind each score.

A score within a group is undefined (``NaN``) when:

- a correlation or ``R^2`` has fewer than two usable pairs;
- a correlation has predictions or response with a range of at most ``1e-6``;
- ``R^2`` has a response with a range of at most ``1e-6``.

Undefined groups are skipped and not scored as zero. The mean is ``NaN`` only if no group is
defined. Error metrics are defined for any group with one pair. A model that predicts a
constant within each drug therefore has no per-drug correlation, whereas its pooled
correlation can be high.

``drevalpy run --group-by drug|sample`` and ``drevalpy aggregate --group-by drug|sample``
print the grouped table after the run. Each cell shows the mean and, in brackets, the mean
number of groups scored; ``undefined`` marks a metric that no group could score.

Normalization against a reference model
---------------------------------------

``ExperimentResult.normalize(reference_model="NaiveMeanEffectsPredictor")`` returns a new
``ExperimentResult`` in which each model is judged against the reference model on the same
fold:

#. For each pair, the reference model's prediction is subtracted from the ground truth and
   from the prediction. A pair the reference never predicted uses ``0.0``.
#. All seven metrics are recomputed on the differences.

The result holds the models other than the reference, and its ``normalized_by`` names the
reference. ``normalize`` raises ``ValueError`` if the result is already normalized, if the reference
model is missing, or if a fold has no reference run.

``drevalpy run`` never adds a reference model; put ``NaiveMeanEffectsPredictor`` in the model
list if you want one. ``drevalpy report`` normalizes only when ``--reference-model`` /
``-r`` is given. Its ``--dataset`` option is accepted but has no effect.

Response transformations
------------------------

A response transformation scales the response before a model is fitted. It is fitted on the
training scope of each fit, and predictions are transformed back, so metrics are computed on
the original response scale.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Name
     - Transformer
   * - ``none``
     - No transformation; the default for ``drevalpy run`` and ``drevalpy single``
   * - ``standard``
     - ``sklearn.preprocessing.StandardScaler``
   * - ``minmax``
     - ``sklearn.preprocessing.MinMaxScaler``
   * - ``robust``
     - ``sklearn.preprocessing.RobustScaler``

The command-line option is ``--response-transformation none|standard|minmax|robust``, matched
without regard to case. In Python, ``drevalpy.run`` and ``drevalpy.single`` take
``response_transformation=`` as an unfitted scikit-learn transformer, which is cloned for each
fit, and ``None`` for no transformation. ``drevalpy.utils.get_response_transformation(name)``
turns ``"standard"``, ``"minmax"`` or ``"robust"`` into such a transformer and ``None`` or
``"None"`` into ``None``; any other name raises ``ValueError``.
