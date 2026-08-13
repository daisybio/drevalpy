"""Metric-name resolution shared by the visualizations.

:meth:`drevalpy.types.results.experiment.ExperimentResult.normalize` recomputes
every metric in :data:`drevalpy.evaluation.AVAILABLE_METRICS` on the residuals
against the reference model and stores them under their **plain** names; the
container records the reference model in ``normalized_by``. That is the contract
the plots are written against.

Results serialized by older versions of drevalpy instead merged an
un-normalized and a normalized metric table into one row, suffixing the
normalized copy with ``": normalized"``. A plot must therefore ask for a metric
by its plain name and let this module decide which key is actually present,
rather than hard-coding either spelling - hard-coding the suffixed one produced
an all-NaN column on the normalized path, which is what crashed the leaderboard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from drevalpy.types.results import ExperimentResult, ModelResult

#: Suffix older drevalpy releases appended to the normalized copy of a metric.
NORMALIZED_SUFFIX = ": normalized"


def metric_keys(result: ExperimentResult | ModelResult) -> set[str]:
    """Collect every metric name any run of *result* reports.

    :param result: Experiment or model result to inspect.
    :returns: Union of the ``metrics`` keys across all runs.
    """
    models = getattr(result, "models", None)
    runs = [run for model in models for run in model.runs] if models is not None else list(result.runs)
    return {name for run in runs for name in run.metrics}


def resolve_metric_key(available: Iterable[str], base: str) -> str | None:
    """Find the key holding *base* among the metric names actually present.

    The plain name wins when both spellings exist: on a normalized experiment it
    already holds the normalized values, and on an un-normalized one it is the
    only honest choice.

    :param available: Metric names present in the result.
    :param base: Plain metric name, for example ``"Pearson"``.
    :returns: The matching key, or ``None`` when the metric is absent.
    """
    names = set(available)
    if base in names:
        return base
    suffixed = f"{base}{NORMALIZED_SUFFIX}"
    return suffixed if suffixed in names else None


def holds_normalized_values(result: ExperimentResult | ModelResult, key: str) -> bool:
    """Whether *key* of *result* holds values normalized against a reference model.

    :param result: Result the metric was read from.
    :param key: Metric key as returned by :func:`resolve_metric_key`.
    :returns: True if the values are normalized.
    """
    return key.endswith(NORMALIZED_SUFFIX) or getattr(result, "normalized_by", None) is not None
