"""Construct AnnData object from flat metrics DataFrame."""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd

_REGULATION_ENCODING = {"up": 1, "down": -1, "not": 0}

_LAYER_METRICS = [
    "EC50",
    "IC50",
    "LN_IC50",
    "AUC",
    "fold_change",
    "slope",
    "front",
    "back",
    "R2",
    "RMSE",
    "p_value",
    "log_p_value",
    "f_value",
    "f_value_sam",
    "relevance_score",
    "signal_quality",
    "regulation",
]


def _pivot_metric(df: pd.DataFrame, metric: str, cell_lines: pd.Index, drugs: pd.Index) -> np.ndarray:
    """Pivot a single metric column into a (cell_lines x drugs) matrix."""
    pivoted = pd.pivot_table(df, values=metric, index="cell_line", columns="drug", aggfunc="first")
    return pivoted.reindex(index=cell_lines, columns=drugs).values


def build_anndata(df: pd.DataFrame) -> anndata.AnnData:
    """Convert a flat metrics DataFrame into an AnnData with (cell_lines x drugs) shape.

    Parameters
    ----------
    df
        DataFrame with columns: cell_line, drug, pEC50, and all other metric columns.

    Returns:
    -------
    AnnData where X is the pEC50 matrix, and all other metrics are stored as layers.
    """
    cell_lines = pd.Index(sorted(df["cell_line"].unique()))
    drugs = pd.Index(sorted(df["drug"].unique()))

    x_matrix = _pivot_metric(df, "pEC50", cell_lines, drugs)

    work_df = df.copy()
    work_df["regulation"] = work_df["regulation"].map(_REGULATION_ENCODING).astype(float)

    layers: dict[str, np.ndarray] = {}
    for metric in _LAYER_METRICS:
        if metric in work_df.columns:
            layers[metric] = _pivot_metric(work_df, metric, cell_lines, drugs)

    return anndata.AnnData(
        X=x_matrix.astype(np.float32),
        obs=pd.DataFrame(index=cell_lines),
        var=pd.DataFrame(index=drugs),
        layers={k: v.astype(np.float32) for k, v in layers.items()},
    )
