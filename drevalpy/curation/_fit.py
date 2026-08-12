from __future__ import annotations

import copy
import math
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


def _build_config(
    n_experiments: int,
    doses: list[float],
    n_replicates: int,
    normalize: bool = False,
    fit_type: str = "OLS",
    fit_speed: str = "exhaustive",
) -> dict:
    """Build a curve_curator config dict equivalent to a parsed TOML."""
    config = {
        "__file__": {"Path": "/tmp/dummy.toml"},  # noqa: S108
        "Meta": {
            "id": "drevalpy",
            "description": "drevalpy curation",
            "condition": "",
            "treatment_time": "72 h",
        },
        "Experiment": {
            "experiments": list(range(n_experiments)),
            "doses": doses,
            "dose_scale": "1e-06",
            "dose_unit": "uM",
            "control_experiment": list(range(n_replicates)),
            "measurement_type": "OTHER",
            "data_type": "OTHER",
            "search_engine": "OTHER",
            "search_engine_version": "0",
        },
        "Paths": {
            "input_file": "/tmp/input.tsv",  # noqa: S108
            "curves_file": "/tmp/curves.tsv",  # noqa: S108
            "normalization_file": "/tmp/norm.txt",  # noqa: S108
            "mad_file": "/tmp/mad.txt",  # noqa: S108
            "dashboard": "/tmp/dashboard.html",  # noqa: S108
        },
        "Processing": {
            "available_cores": 1,
            "max_missing": max(len(doses) - 5, 0),
            "imputation": False,
            "normalization": normalize,
        },
        "Curve Fit": {
            "type": fit_type,
            "speed": fit_speed,
            "max_iterations": 1000,
            "interpolation": False,
            "control_fold_change": True,
        },
        "F Statistic": {
            "optimized_dofs": True,
            "alpha": 0.05,
            "fc_lim": 0.45,
        },
    }

    from curve_curator.toml_parser import set_default_values

    config = set_default_values(config)
    return config


def _fit_chunk(chunk_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Fit a single chunk using run_pipeline (single-core)."""
    from curve_curator import quantification

    cfg = copy.deepcopy(config)
    cfg["Processing"]["available_cores"] = 1
    return quantification.run_pipeline(chunk_df, cfg)


def fit_groups(
    groups: list[tuple[pd.DataFrame, dict]],
    cores: int,
    normalize: bool = False,
    fit_type: str = "OLS",
    fit_speed: str = "exhaustive",
) -> list[tuple[pd.DataFrame, dict]]:
    """Fit all groups with chunk-level parallelism.

    Parameters
    ----------
    groups
        List of (wide_df, group_info) from preprocess.
    cores
        Number of CPU cores for parallel fitting.
    normalize
        Whether to apply median-centric normalization.
    fit_type
        Fitting method: "OLS" or "MLE".
    fit_speed
        Fitting thoroughness: "fast", "standard", "exhaustive", or "basinhopping".

    Returns:
    -------
    List of (fitted_df, config) tuples.
    """
    total_curves = sum(len(df) for df, _ in groups)
    chunk_size = max(1, total_curves // cores)

    work_items: list[tuple[pd.DataFrame, dict, int]] = []
    configs: list[dict] = []

    for group_idx, (df, group_info) in enumerate(groups):
        config = _build_config(
            n_experiments=group_info["n_experiments"],
            doses=group_info["doses"],
            n_replicates=group_info["n_replicates"],
            normalize=normalize,
            fit_type=fit_type,
            fit_speed=fit_speed,
        )
        configs.append(config)

        n_chunks = math.ceil(len(df) / chunk_size)
        for chunk_df in np.array_split(df, n_chunks):
            chunk_df = chunk_df.reset_index(drop=True)
            work_items.append((chunk_df, config, group_idx))

    fitted_chunks: list[tuple[pd.DataFrame, int]] = []

    if cores <= 1 or len(work_items) == 1:
        for chunk_df, config, group_idx in work_items:
            result = _fit_chunk(chunk_df, config)
            fitted_chunks.append((result, group_idx))
    else:
        with ProcessPoolExecutor(max_workers=cores) as executor:
            futures = [
                (executor.submit(_fit_chunk, chunk_df, config), group_idx)
                for chunk_df, config, group_idx in work_items
            ]
            for future, group_idx in futures:
                fitted_chunks.append((future.result(), group_idx))

    n_groups = len(groups)
    group_results: list[list[pd.DataFrame]] = [[] for _ in range(n_groups)]
    for fitted_df, group_idx in fitted_chunks:
        group_results[group_idx].append(fitted_df)

    from curve_curator import thresholding

    results: list[tuple[pd.DataFrame, dict]] = []
    for group_idx in range(n_groups):
        assembled = pd.concat(group_results[group_idx], ignore_index=True)
        assembled = thresholding.apply_significance_thresholds(assembled, configs[group_idx])
        results.append((assembled, configs[group_idx]))

    return results
