"""The leaderboard table: one row per model with the four ranked metrics.

The columns follow the 1.6.1 ``create_leaderboard.load_results`` table:

* ``PCC``: the Pearson correlation of the predictions, normalized when the experiment was normalized
  against a reference model;
* ``RMSE``: the root mean square error of the unnormalized predictions;
* ``Pearson``: the raw (unnormalized) pooled Pearson correlation;
* ``Pearson_per_drug``: the Pearson correlation within each drug, averaged over the drugs where it is
  defined (see :mod:`drevalpy._grouped_evaluation`) and then over folds.

Each has a ``_std`` column with the sample standard deviation (``ddof=1``) over folds, ``0`` for a
single fold. After :meth:`~drevalpy.types.results.ExperimentResult.normalize` the unnormalized
numbers come from ``RunResult.raw_metrics`` and ``RunResult.raw_per_drug_pearson``, because the
runs themselves then hold residuals. ``pandas`` is imported inside the function, like ``matplotlib``
in the drawing helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy._grouped_evaluation import run_group_score

if TYPE_CHECKING:
    import pandas as pd

    from drevalpy.types.results import ExperimentResult
    from drevalpy.types.results.run import RunResult

#: Metric columns of the table, each followed by its ``_std`` column.
METRIC_COLUMNS = ("PCC", "RMSE", "Pearson", "Pearson_per_drug")

_NAN = float("nan")


def _run_row(run: RunResult, normalized: bool) -> dict:
    """Read the four leaderboard metrics of one run.

    :param run: Ordinary run of a model, or the summary of the reference model.
    :param normalized: Whether the experiment was normalized against a reference model.
    :returns: The algorithm name and one value per entry of :data:`METRIC_COLUMNS`, NaN where unavailable.
    """
    if run.raw_metrics:
        raw, per_drug = run.raw_metrics, run.raw_per_drug_pearson
    elif normalized:
        raw, per_drug = {}, None
    else:
        raw, per_drug = run.metrics, run_group_score(run, "drug", "Pearson")
    return {
        "algorithm": run.model_name,
        "PCC": run.metrics.get("Pearson", _NAN),
        "RMSE": raw.get("RMSE", run.metrics.get("RMSE", _NAN)),
        "Pearson": raw.get("Pearson", _NAN),
        "Pearson_per_drug": _NAN if per_drug is None else per_drug,
    }


def build_leaderboard_df(result: ExperimentResult) -> pd.DataFrame:
    """Build the aggregated leaderboard table from an ExperimentResult.

    :param result: Experiment to rank. The reference model of a normalized experiment is included
        as a baseline row, as in 1.6.1.
    :returns: One row per model with the mean and standard deviation of every metric over folds, the
        ``is_baseline`` flag (names starting with ``Naive``), sorted by ``PCC`` descending.
    """
    import pandas as pd

    normalized = result.normalized_by is not None
    runs = [run for model in result.models for run in model.select_runs()] + list(result.reference_runs)
    columns = ["algorithm"]
    for name in METRIC_COLUMNS:
        columns += [name, f"{name}_std"]
    df = pd.DataFrame([_run_row(run, normalized) for run in runs], columns=["algorithm", *METRIC_COLUMNS])
    if df.empty:
        return pd.DataFrame(columns=[*columns, "is_baseline"])

    df_agg = df.groupby("algorithm").agg(dict.fromkeys(METRIC_COLUMNS, ["mean", "std"])).reset_index()
    df_agg.columns = columns
    for name in METRIC_COLUMNS:
        df_agg[f"{name}_std"] = df_agg[f"{name}_std"].fillna(0)
    df_agg["is_baseline"] = df_agg["algorithm"].str.startswith("Naive")
    return df_agg.sort_values("PCC", ascending=False).reset_index(drop=True)
