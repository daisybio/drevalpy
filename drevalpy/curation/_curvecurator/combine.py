"""Combine fitted CurveCurator results into a drevalpy dataset table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from drevalpy.curation._curvecurator.types import CurationFitResult
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER

CURVECURATOR_OUTPUT_COLUMNS = {
    "Name": "Name",
    "pEC50": "pEC50_curvecurator",
    "pEC50 Error": "pEC50Error",
    "Curve Slope": "Slope",
    "Curve Front": "Front",
    "Curve Back": "Back",
    "Curve Fold Change": "FoldChange",
    "Curve AUC": "AUC_curvecurator",
    "Curve R2": "R2",
    "Curve P_Value": "pValue",
    "Curve Relevance Score": "RelevanceScore",
    "Curve F_Value": "fValue",
    "Curve Log P_Value": "negLog10pValue",
    "Signal Quality": "SignalQuality",
    "Curve RMSE": "RMSE",
    "Curve F_Value SAM Corrected": "fValueSAMCorrected",
    "Curve Regulation": "Regulation",
}


def _calc_ic50(model_params_df: pd.DataFrame) -> None:
    """Add IC50 and LN_IC50 columns derived from fitted CurveCurator parameters."""

    def ic50(front, back, slope, pec50):
        with np.errstate(invalid="ignore"):
            return np.power(10, (np.log10((front - 0.5) / (0.5 - back)) - slope * pec50) / slope)

    front = model_params_df["Front"].values
    back = model_params_df["Back"].values
    slope = model_params_df["Slope"].values
    pec50 = model_params_df["pEC50_curvecurator"].values - 6

    model_params_df["IC50_curvecurator"] = ic50(front, back, slope, pec50)
    model_params_df["LN_IC50_curvecurator"] = np.log(model_params_df["IC50_curvecurator"].values)


def _normalize_curves_table(curves: pd.DataFrame) -> pd.DataFrame:
    available = {source: target for source, target in CURVECURATOR_OUTPUT_COLUMNS.items() if source in curves.columns}
    fitted_curve_data = curves.loc[:, list(available)].rename(columns=available).copy()
    fitted_curve_data[[CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER]] = fitted_curve_data.Name.str.split("|", expand=True)
    fitted_curve_data["EC50_curvecurator"] = np.power(10, -fitted_curve_data["pEC50_curvecurator"].values) * 10**6
    _calc_ic50(fitted_curve_data)
    return fitted_curve_data


def combine(
    fit_results: tuple[CurationFitResult, ...] | list[CurationFitResult],
) -> pd.DataFrame:
    """Combine fitted CurveCurator results into one drevalpy dataset table.

    :param fit_results: Fitted CurveCurator results.
    :returns: Combined curated dataset table.
    :raises ValueError: If no fit results are provided.
    """
    if not fit_results:
        raise ValueError("At least one CurveCurator fit result is required.")

    tables = [_normalize_curves_table(result.curves) for result in fit_results if not result.curves.empty]
    if not tables:
        return pd.DataFrame()

    combined = pd.concat(tables, ignore_index=True)
    combined.index = combined["Name"]
    return combined


def write_dataset_csv(dataset: pd.DataFrame, output_file: str | Path) -> Path:
    """Write a combined dataset table to CSV.

    :param dataset: Combined curated dataset table.
    :param output_file: Destination CSV path.
    :returns: Written CSV path.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=True)
    return output_path
