"""Extract and rename curve metrics from curve_curator output, derive IC50/EC50."""

from __future__ import annotations

import numpy as np
import pandas as pd

_COLUMN_RENAME = {
    "pEC50": "pEC50",
    "Curve Slope": "slope",
    "Curve Front": "front",
    "Curve Back": "back",
    "Curve Fold Change": "fold_change",
    "Curve AUC": "AUC",
    "Curve RMSE": "RMSE",
    "Curve R2": "R2",
    "Curve P_Value": "p_value",
    "Curve Log P_Value": "log_p_value",
    "Curve F_Value": "f_value",
    "Curve F_Value SAM Corrected": "f_value_sam",
    "Curve Relevance Score": "relevance_score",
    "Curve Regulation": "regulation",
    "Signal Quality": "signal_quality",
    # Per-parameter standard errors. CurveCurator computes these on every fit at
    # every speed, from a Moore-Penrose pseudo-inverse of the Jacobian
    # (``models.LogisticModel.calculate_parameter_error``), and emits them in
    # ``quantification.add_logistic_model``'s ``fit_cols``. They are the only
    # per-curve uncertainty estimate the pipeline produces, so they are kept.
    "pEC50 Error": "pec50_error",
    "Curve Slope Error": "slope_error",
    "Curve Front Error": "front_error",
    "Curve Back Error": "back_error",
}

_LABEL_COLUMNS = ["cell_line", "drug"]

#: Metrics derived here rather than read from the fit.
DERIVED_METRICS = ("EC50", "IC50", "LN_IC50")

_KEEP_COLUMNS = [*_LABEL_COLUMNS, *_COLUMN_RENAME.values(), *DERIVED_METRICS]


def _compute_ic50(front: np.ndarray, back: np.ndarray, slope: np.ndarray, pec50: np.ndarray) -> np.ndarray:
    """Compute IC50 in uM from fitted curve parameters using closed-form solution."""
    with np.errstate(invalid="ignore"):
        pec50_um = pec50 - 6
        return np.power(10, (np.log10((front - 0.5) / (0.5 - back)) - slope * pec50_um) / slope)


def postprocess(groups: list[tuple[pd.DataFrame, dict]]) -> pd.DataFrame:
    """Extract/rename metrics from curve_curator output and compute derived metrics.

    Parameters
    ----------
    groups
        Fitted DataFrames from curve_curator (one per dose-range group),
        each paired with its config dict.

    Returns:
    -------
    Single DataFrame with columns: cell_line, drug, plus all metric columns.
    All curves are preserved (no filtering). The label columns hold whatever
    ``cell_line``/``drug`` values were fitted, so native identifiers survive the
    round trip - as strings, because they travel through curve_curator's
    ``Name`` column.

    Raises:
    ------
    KeyError
        If a fitted frame is missing a metric curve_curator is expected to emit.
        Silently dropping one is what lost the per-curve errors for a year.
    """
    frames: list[pd.DataFrame] = []

    for fitted_df, _config in groups:
        _require_metric_columns(fitted_df)
        df = fitted_df.copy()

        df[_LABEL_COLUMNS] = df["Name"].str.split("|", expand=True)

        df = df.rename(columns=_COLUMN_RENAME)

        df["EC50"] = np.power(10, -df["pEC50"].values) * 1e6

        df["IC50"] = _compute_ic50(
            front=df["front"].values,
            back=df["back"].values,
            slope=df["slope"].values,
            pec50=df["pEC50"].values,
        )
        df["LN_IC50"] = np.log(df["IC50"].values)

        frames.append(df[_KEEP_COLUMNS])

    return pd.concat(frames, ignore_index=True)


def _require_metric_columns(fitted_df: pd.DataFrame) -> None:
    """Fail loudly when curve_curator did not emit a metric we rename."""
    missing = [column for column in _COLUMN_RENAME if column not in fitted_df.columns]
    if missing:
        raise KeyError(f"curve_curator output is missing expected metric column(s): {missing}")
