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
With ``group_by="drug"``, ``"sample"`` or ``"study"`` each metric is computed within every
drug, sample or study of a fold, averaged over the groups where it is defined, and then
averaged over folds. Any other ``group_by`` value raises ``ValueError``.
``summary_group_counts(group_by)`` returns the mean number of groups behind each score.

The study of a pair is the part of its sample ID before ``::``, as written by
``drevalpy.data.concat``. A pair whose sample ID has no such prefix belongs to no study, so
``group_by="study"`` on a run of a single dataset has no groups and its score is ``NaN``.
Under the ``LStO`` split mode every fold has one test study, so the per-study score equals the
pooled score; the grouping is informative with the other split modes on a concatenated dataset,
whose test folds mix the studies. The score of each ``LStO`` fold is in
``run.metrics`` and the study in ``run.fold_metadata["test_study"]``; see
:doc:`/how-to/run_cross_study`.

A score within a group is undefined (``NaN``) when:

- a correlation or ``R^2`` has fewer than two usable pairs;
- a correlation has predictions or response with a range of at most ``1e-6``;
- ``R^2`` has a response with a range of at most ``1e-6``.

Undefined groups are skipped and not scored as zero. The mean is ``NaN`` only if no group is
defined. Error metrics are defined for any group with one pair. A model that predicts a
constant within each drug therefore has no per-drug correlation, whereas its pooled
correlation can be high.

A per-group ``R^2`` is unbounded below, and a group is scored whenever it has two pairs and a
response range above ``1e-6``. A group with few test pairs and a nearly flat response, for
example 2 pairs with a response range of 0.01, can score an ``R^2`` in the tens of thousands
below zero. Such a group enters the mean over groups with the same weight as any other, so a
few of them can dominate it. The mean number of groups behind each score, from
``summary_group_counts``, does not show how small the groups are.

``drevalpy run --group-by drug|sample|study`` and ``drevalpy aggregate --group-by drug|sample|study``
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
   * - ``drug_mean``
     - ``drevalpy.utils.response_transform.GroupMeanCenterer(("drug",))``
   * - ``drug_tissue_mean``
     - ``drevalpy.utils.response_transform.GroupMeanCenterer(("drug", "tissue"))``

The command-line option is ``--response-transformation none|standard|minmax|robust|drug_mean|drug_tissue_mean``, matched
without regard to case. In Python, ``drevalpy.run`` and ``drevalpy.single`` take
``response_transformation=`` as an unfitted scikit-learn transformer, which is cloned for each
fit, and ``None`` for no transformation. ``drevalpy.utils.get_response_transformation(name)``
turns ``"standard"``, ``"minmax"``, ``"robust"``, ``"drug_mean"`` or ``"drug_tissue_mean"`` into
such a transformer and ``None`` or ``"None"`` into ``None``; any other name raises ``ValueError``.

``drug_mean`` and ``drug_tissue_mean`` subtract a group mean instead of rescaling. The means come
from the training pairs of the fit, and the same group mean is added back to each prediction.
``drug_tissue_mean`` uses the mean of the pair's (drug, tissue) combination, falls back to the
mean of its drug when that combination was not in the training pairs, and to the global training
mean when the drug is unseen as well. The drug comes from the pair and the tissue from the sample
metadata of the dataset being fitted or predicted, so ``drug_tissue_mean`` needs a ``tissue``
sample column. For a single-drug model the offset of a drug is a constant shift of that drug's own
responses. ``drevalpy run`` always fits ``NaiveMeanEffectsPredictor``, the reference of the
normalized metrics, on the untransformed response.
