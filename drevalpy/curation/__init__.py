"""Dose-response curve fitting via curve_curator, returning AnnData."""

from __future__ import annotations

import anndata
import pandas as pd

from drevalpy.curation._anndata import build_anndata
from drevalpy.curation._fit import fit_groups
from drevalpy.curation._postprocess import postprocess
from drevalpy.curation._preprocess import preprocess


def curate(
    df: pd.DataFrame,
    *,
    cores: int = 4,
    normalize: bool = False,
    fit_type: str = "OLS",
    fit_speed: str = "exhaustive",
) -> anndata.AnnData:
    """Fit dose-response curves and return an AnnData of curve metrics.

    Parameters
    ----------
    df
        Long-form DataFrame with columns: drug, cell_line, concentration,
        intensity, and optionally replicate.
    cores
        Number of CPU cores for parallel fitting.
    normalize
        Whether to apply median-centric normalization before fitting.
    fit_type
        Fitting method: "OLS" or "MLE".
    fit_speed
        Fitting thoroughness: "fast", "standard", "exhaustive", or "basinhopping".

    Returns:
    -------
    AnnData with shape (n_cell_lines, n_drugs). X contains pEC50 values.
    All curve metrics are stored as layers. No quality filtering is applied.
    """
    groups = preprocess(df)
    fitted_groups = fit_groups(
        groups,
        cores=cores,
        normalize=normalize,
        fit_type=fit_type,
        fit_speed=fit_speed,
    )
    metrics_df = postprocess(fitted_groups)
    adata = build_anndata(metrics_df)
    return adata
