from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = ["drug", "cell_line", "concentration", "intensity"]


def preprocess(df: pd.DataFrame) -> list[tuple[pd.DataFrame, dict]]:
    """Validate, group by dose range, pivot to wide-form, and return (df, group_info) tuples.

    Parameters
    ----------
    df
        Long-form DataFrame with columns: drug, cell_line, concentration,
        intensity, and optionally replicate.

    Returns:
    -------
    List of (wide_df, group_info) tuples where group_info contains
    n_experiments, doses, n_replicates for building the config later.
    """
    _validate_columns(df)
    df = df.copy()

    has_replicate = "replicate" in df.columns

    df["mindose"] = df.groupby(["cell_line", "drug"])["concentration"].transform("min")
    df["maxdose"] = df.groupby(["cell_line", "drug"])["concentration"].transform("max")

    groupby: list[str] = []
    if df["maxdose"].nunique() > 1:
        groupby.append("maxdose")
    if df["mindose"].nunique() > 1:
        groupby.append("mindose")
    if has_replicate:
        df["nreplicates"] = df.groupby(["cell_line", "drug"])["replicate"].transform("nunique")
        if df["nreplicates"].nunique() > 1:
            groupby.append("nreplicates")

    if groupby:
        groups = df.groupby(groupby)
    else:
        groups = [("all", df)]

    results: list[tuple[pd.DataFrame, dict]] = []
    for _index, group_df in groups:
        wide_df, group_info = _prepare_wide(group_df, has_replicate)
        results.append((wide_df, group_info))

    return results


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _prepare_wide(
    curve_df: pd.DataFrame,
    has_replicate: bool,
) -> tuple[pd.DataFrame, dict]:
    """Pivot long-form to wide-form with synthetic control columns."""
    if has_replicate:
        n_replicates = curve_df["replicate"].nunique()
        pivot_columns = ["concentration", "replicate"]
    else:
        n_replicates = 1
        pivot_columns = ["concentration"]

    dup_subset = ["cell_line", "drug"] + pivot_columns
    if curve_df.duplicated(subset=dup_subset).any():
        warnings.warn(
            f"Duplicate entries found for {tuple(dup_subset)} combinations. Aggregating using mean of 'intensity'.",
            UserWarning,
            stacklevel=2,
        )
        curve_df = curve_df.groupby(dup_subset, as_index=False)["intensity"].mean()

    wide = curve_df.pivot(index=["cell_line", "drug"], columns=pivot_columns, values="intensity")

    if has_replicate:
        control_df = pd.DataFrame(
            {(0.0, col_id): 1.0 for col_id in range(n_replicates)},
            index=wide.index,
        )
    else:
        control_df = pd.DataFrame({0.0: 1.0}, index=wide.index)

    wide = pd.concat([control_df, wide], axis=1)

    concentrations = wide.columns.sort_values()
    doses = concentrations.get_level_values(0).to_list()
    wide = wide[concentrations]

    n_experiments = wide.shape[1]
    experiments = np.arange(n_experiments)
    wide.insert(0, "Name", ["|".join(map(str, idx)) for idx in wide.index.tolist()])

    wide.columns = ["Name"] + [f"Raw {i}" for i in experiments]
    wide = wide.reset_index(drop=True)

    group_info = {
        "n_experiments": n_experiments,
        "doses": doses,
        "n_replicates": n_replicates,
    }
    return wide, group_info
