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
}

_KEEP_COLUMNS = ["cell_line", "drug", *_COLUMN_RENAME.values(), "EC50", "IC50", "LN_IC50"]


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
    All curves are preserved (no filtering).
    """
    frames: list[pd.DataFrame] = []

    for fitted_df, _config in groups:
        df = fitted_df.copy()

        df[["cell_line", "drug"]] = df["Name"].str.split("|", expand=True)

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
