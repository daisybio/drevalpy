Metrics and response transformations
====================================

:doc:`/explanation/metrics` explains why pooled scores can mislead and why the per-drug
metrics and a naive baseline matter. This page lists the metrics and how they are computed.

Metrics
-------

``drevalpy.evaluation.Metric`` lists the metrics. Each member is its name, so
``Metric.PEARSON == "Pearson"``, and a name works wherever a metric is expected.
``Metric(name)`` looks a name up and raises ``ValueError`` listing the names for an unknown one;
iterating ``Metric`` gives every metric in the order of the table below. The names are
case-sensitive and ``R^2`` (``Metric.R2``) is written with a caret. Each metric has a
``.kind`` (``MetricKind.ERROR``, ``FIT`` or ``CORRELATION``), which decides when its score is
undefined, and a ``.direction``, ``"min"`` for an error and ``"max"`` otherwise.

Each metric is computed by the scipy or scikit-learn function in the last column
(``metric.function.path``), called on the pairs of each group; a pooled score is the metric over
a single group. DrEvalPy decides
only which groups are scored (see `Missing and degenerate input`_), so the function never sees a
degenerate group.

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Name
     - Better when
     - Computed by
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

Entry points
~~~~~~~~~~~~

``evaluate(predictions, response, metrics=None)``
   Returns the pooled score of each metric as a ``dict`` keyed by name. ``metrics`` is ``None``
   for every metric, one name, or several names.

``evaluate_per_group(predictions, response, metrics=None, *, labels, groups=None)``
   Scores each metric within every group of pairs, where ``labels`` holds the group of each pair
   (a NaN label puts the pair in no group). It returns a ``PerGroupScores``: the sorted
   ``groups``, the number of pairs ``n_pairs`` and the score ``values[name]`` of each group, and
   ``mean()`` and ``n_defined()``, the mean over the groups where a score is defined and how many
   they are. ``groups=`` fixes the group axis, so several calls line up; a group without pairs
   then scores ``NaN``.

Both take one-dimensional arrays of equal length and raise ``ValueError`` for an unknown
metric.

.. code-block:: python

   from drevalpy.evaluation import Metric, evaluate, evaluate_per_group

   evaluate(predictions, response, ["Pearson", "RMSE"])
   per_drug = evaluate_per_group(predictions, response, Metric.PEARSON, labels=drug_ids)
   per_drug.mean()["Pearson"]

Missing and degenerate input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pair is dropped if the prediction or the response is NaN. A score is undefined, and is
``NaN``, when the data cannot support the metric. A correlation with constant predictions
is not undefined: the model has no ranking skill, and the score is ``0.0``. Then:

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - Situation
     - Score
   * - No pair, for ``MSE``, ``RMSE`` or ``MAE``
     - ``NaN``
   * - Fewer than two pairs, for ``R^2`` or a correlation
     - ``NaN``
   * - Constant response (within ``1e-6``), for ``R^2`` or a correlation
     - ``NaN``
   * - Constant predictions (within ``1e-6``) and a varying response, for a correlation
     - ``0.0``
   * - Constant predictions and a varying response, for ``R^2``
     - ``1 - SSE/SST``, as for any other predictions

The response is tested first, so a correlation with both sides constant is ``NaN``. The
same rule applies to a pooled score and to the score within a group. Two cases differ from
DrEvalPy 1.x: a pooled ``R^2`` on a constant response is ``NaN`` (scikit-learn returns
``0.0`` or ``1.0``), and a pooled correlation with both sides constant is ``NaN`` (1.x
returned ``0.0``).

Pooled and per-group scores
---------------------------

By default, a fold's metric is pooled: it is computed over all of the fold's pairs, and the
result is the mean over folds.

``ExperimentResult.summary_table(*, group_by=None, dataset=None)`` returns a
``pandas.DataFrame`` with one row per model and one column per metric. With a ``group_by``
each metric is computed within every group of a fold, averaged over the groups where it is
defined, and then averaged over folds. ``summary_group_counts(group_by, dataset=None)``
returns the mean number of groups behind each score. The accepted ``group_by`` values are:

- ``"drug"``, ``"sample"`` and ``"study"``, read from the IDs of the pairs;
- the name of a sample annotation column of the dataset (``dataset.sample_meta``), such as
  ``"tissue"`` or ``"cancer_type"``, or of a drug annotation column (the variables of
  ``dataset.response``), such as ``"ncit_class"``, ``"target_moa"``, ``"target_gene"``,
  ``"target_class_l1"`` to ``"target_class_l5"`` or ``"drug_role"``. Pass the dataset the runs
  were made on as ``dataset=``;
- ``"sample:<column>"`` or ``"drug:<column>"`` to choose the axis explicitly. Use this when a
  column name exists on both axes.

.. code-block:: python

   results.summary_table(group_by="tissue", dataset=dataset)

A pair belongs to no group, and is left out of the score, when its label is missing, empty or
``"unknown"`` (in any case), or when its sample or drug ID is not in ``dataset``. ``ValueError``
is raised for a bare column name that exists on both axes, for a name that is neither built in
nor a column of the dataset, and for an annotation grouping without ``dataset=``. A ``dataset``
passed with ``"drug"``, ``"sample"`` or ``"study"`` is not read.

A per-tissue or per-drug-class score pools many drugs (or samples), so, like a pooled score, it
still rewards a model that only knows which drugs are potent. Only ``"drug"`` scores the ranking of
samples within one drug.

The study of a pair is the part of its sample ID before ``::``, as written by
``drevalpy.data.concat``. A pair whose sample ID has no such prefix belongs to no study, so
``group_by="study"`` on a run of a single dataset has no groups and its score is ``NaN``.
Under the ``LStO`` split mode every fold has one test study, so the per-study score equals the
pooled score; the grouping is informative with the other split modes on a concatenated dataset,
whose test folds mix the studies. The score of each ``LStO`` fold is in
``run.metrics`` and the study in ``run.fold_metadata["test_study"]``; see
:doc:`/how-to/run_cross_study`.

A score within a group follows the rule above. It is undefined (``NaN``) when:

- an error metric has no usable pair;
- a correlation or ``R^2`` has fewer than two usable pairs;
- a correlation or ``R^2`` has a response with a range of at most ``1e-6``.

These depend on the data only, so the same groups are undefined for every model on the same
pairs. Undefined groups are skipped in the mean. The mean is ``NaN`` only if no group is
defined. A correlation of constant predictions within a group is not undefined: it is
``0.0`` and counts in the mean. A model that predicts a constant within each drug, such as
``NaiveDrugMeanPredictor``, therefore scores a per-drug correlation of ``0.0``, whereas its
pooled correlation can be high.

A per-group ``R^2`` is unbounded below, and a group is scored whenever it has two pairs and a
response range above ``1e-6``. A group with few test pairs and a nearly flat response, for
example 2 pairs with a response range of 0.01, can score an ``R^2`` in the tens of thousands
below zero. Such a group enters the mean over groups with the same weight as any other, so a
few of them can dominate it. The mean number of groups behind each score, from
``summary_group_counts``, does not show how small the groups are.

``drevalpy run --group-by <grouping>`` and ``drevalpy aggregate --group-by <grouping>``
print the grouped table after the run, where ``<grouping>`` is ``drug``, ``sample``, ``study`` or
an annotation column as above. ``run`` reads the annotation from the dataset of the run.
``aggregate`` does not know the dataset, so an annotation grouping needs
``--dataset NAME_OR_PATH`` (``-d``), the registered name or ``.h5mu`` path of the dataset the runs
were made on. Repeat the option for runs on several datasets; they are combined as in
``run``. ``--dataset`` is not read for ``drug``, ``sample`` and ``study``. Each cell shows the mean and, in brackets, the mean
number of groups scored; ``undefined`` marks a metric that no group could score, which can
only happen because of the data, so it is the same for every model.

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
     - ``drevalpy.models.response_transformation.GroupMeanCenterer(("drug",))``
   * - ``drug_tissue_mean``
     - ``drevalpy.models.response_transformation.GroupMeanCenterer(("drug", "tissue"))``

The command-line option is ``--response-transformation none|standard|minmax|robust|drug_mean|drug_tissue_mean``, matched
without regard to case. In Python, ``drevalpy.run`` and ``drevalpy.single`` take
``response_transformation=`` as an unfitted scikit-learn transformer, which is cloned for each
fit, and ``None`` for no transformation. ``drevalpy.models.get_response_transformation(name)``
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

.. autoclass:: drevalpy.models.response_transformation.GroupMeanCenterer
   :members:
   :exclude-members: set_fit_request, set_transform_request, set_inverse_transform_request
   :show-inheritance:
   :member-order: bysource
