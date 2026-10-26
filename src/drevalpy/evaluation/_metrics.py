"""The evaluation metrics: what each one measures, its library function, and when its score is undefined.

:class:`Metric` is the one list of metrics. Each member is its own name (a ``str``, the name results
are stored under), carries its :class:`MetricKind`, and calls one ``scipy.stats`` or
``sklearn.metrics`` function. drevalpy owns no formula.

drevalpy owns the **undefined-score rule**, keyed on the kind and written once, in
:func:`score_pairs`. It applies to a pooled score (all pairs as one group) and to the score within
each group alike:

* **The data cannot support the score:** no pair, or for ``R^2`` and the correlations fewer than
  :data:`MIN_GROUP_SIZE` pairs or a constant response. The score is ``NaN``, the same for every
  model scored on these pairs.
* **The model makes a correlation undefined:** a constant prediction while the response varies.
  The model has no ranking skill there, and the correlation scores ``0.0``.
* Otherwise, including ``R^2`` of a constant prediction, the library function decides.

The response is tested before the prediction, so both sides constant is ``NaN``. "Constant" is
:func:`~drevalpy.evaluation._pairs.is_constant`. The library is never called on a group the rule
decides, so it never warns about constant input or returns ``r2_score``'s ``0.0``/``1.0`` for a
flat response.

A new metric is one new member: its stored name, its kind and its :class:`LibraryFunction`. The
CLI choices, the plots and the scoring functions read the enum; its row in
``docs/reference/metrics.rst`` is written by hand. A member's value is the name saved results use,
so existing values stay as they are. On Python 3.11, ``"x" in Metric`` raises ``TypeError``; test a
name with ``name in {metric.value for metric in Metric}``.

``scipy.stats`` and ``sklearn.metrics`` are imported on the first call of a function: this module
is on the import path of ``import drevalpy`` (see ``tests/test_import_cost_policy.py``).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Final, Literal, NoReturn

if TYPE_CHECKING:
    import numpy as np

    from ._pairs import Pairs

#: ``R^2`` and the correlations are undefined below this many pairs.
MIN_GROUP_SIZE: Final[int] = 2

#: Whether lower (``"min"``) or higher (``"max"``) values of a metric are better.
Direction = Literal["min", "max"]


class MetricKind(Enum):
    """What a metric measures. The kind decides when the data supports a score and which way is better."""

    #: A prediction error: ``MSE``, ``RMSE``, ``MAE``. Lower is better; defined with one pair.
    ERROR = "error"
    #: The goodness of fit: ``R^2``. Higher is better; needs two pairs and a varying response.
    FIT = "fit"
    #: A correlation: ``Pearson``, ``Spearman``, ``Kendall``. As fit, and a constant prediction scores ``0.0``.
    CORRELATION = "correlation"


@dataclass(frozen=True)
class LibraryFunction:
    """A ``scipy.stats`` or ``sklearn.metrics`` function, imported on its first call.

    It owns the two library conventions, so no metric repeats them: scikit-learn takes
    ``(y_true, y_pred)``, and scipy returns a result whose ``statistic`` is the score.
    """

    #: Module of the function, ``"scipy.stats"`` or ``"sklearn.metrics"``.
    module: Literal["scipy.stats", "sklearn.metrics"]
    #: Name of the function in :attr:`module`.
    name: str

    @property
    def path(self) -> str:
        """Return the dotted path, for example ``"sklearn.metrics.r2_score"``."""
        return f"{self.module}.{self.name}"

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Score one group's pairs.

        :param y_pred: Predictions.
        :param y_true: Observed responses.
        :returns: The library's score.
        """
        function = getattr(importlib.import_module(self.module), self.name)
        if self.module == "sklearn.metrics":
            return float(function(y_true, y_pred))
        return float(function(y_pred, y_true).statistic)


class Metric(StrEnum):
    """The evaluation metrics, in report order. Each member is its name: ``Metric.PEARSON == "Pearson"``.

    The names are case-sensitive and are the keys results are stored under. ``Metric(name)`` looks a
    name up and raises ``ValueError`` listing the names for an unknown one; iterate ``Metric`` for every
    metric. Score with :func:`drevalpy.evaluation.evaluate` or :func:`drevalpy.evaluation.evaluate_per_group`.
    """

    #: What the metric measures; decides its undefined scores and its :attr:`direction`.
    kind: MetricKind
    #: The library function, called only on pairs the rule leaves to it.
    function: LibraryFunction

    def __new__(cls, name: str, kind: MetricKind, function: LibraryFunction) -> Metric:
        """Create a member from its row below.

        :param name: The stored name.
        :param kind: What the metric measures.
        :param function: The library function.
        :returns: The member, equal to and hashing like ``name``.
        """
        member = str.__new__(cls, name)
        member._value_ = name
        member.kind = kind
        member.function = function
        return member

    MSE = "MSE", MetricKind.ERROR, LibraryFunction("sklearn.metrics", "mean_squared_error")
    RMSE = "RMSE", MetricKind.ERROR, LibraryFunction("sklearn.metrics", "root_mean_squared_error")
    MAE = "MAE", MetricKind.ERROR, LibraryFunction("sklearn.metrics", "mean_absolute_error")
    R2 = "R^2", MetricKind.FIT, LibraryFunction("sklearn.metrics", "r2_score")
    PEARSON = "Pearson", MetricKind.CORRELATION, LibraryFunction("scipy.stats", "pearsonr")
    SPEARMAN = "Spearman", MetricKind.CORRELATION, LibraryFunction("scipy.stats", "spearmanr")
    KENDALL = "Kendall", MetricKind.CORRELATION, LibraryFunction("scipy.stats", "kendalltau")

    @classmethod
    def _missing_(cls, value: object) -> NoReturn:
        """Reject an unknown name with the list of known ones.

        :param value: The name that matched no member.
        :raises ValueError: Always.
        """
        raise ValueError(f"Unknown metric {value!r}. Available: {', '.join(cls)}")

    @property
    def direction(self) -> Direction:
        """Whether lower or higher is better: ``"min"`` for an error, ``"max"`` for ``R^2`` and the correlations."""
        return "min" if self.kind is MetricKind.ERROR else "max"


def score_pairs(metric: Metric, pairs: Pairs) -> float:
    """Score one group under the undefined-score rule of the module docstring.

    :param metric: The metric.
    :param pairs: The pairs of one group.
    :returns: ``NaN`` where the data cannot support the score, ``0.0`` for a correlation of a
        constant prediction, otherwise the library function's score.
    """
    if len(pairs) == 0:
        return float("nan")
    if metric.kind is not MetricKind.ERROR:
        if len(pairs) < MIN_GROUP_SIZE or pairs.response_is_constant:
            return float("nan")
        if metric.kind is MetricKind.CORRELATION and pairs.prediction_is_constant:
            return 0.0
    return metric.function(pairs.y_pred, pairs.y_true)
