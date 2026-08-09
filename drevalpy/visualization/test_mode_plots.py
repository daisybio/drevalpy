"""Test-mode plotting orchestration for visualization reports."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from upath import UPath as Path

from . import ComparisonScatter, CrossStudyTables, Heatmap, Violin
from .critical_difference_plot import CriticalDifferencePlot


def _require_prediction_subset(ev_res: pd.DataFrame, test_mode: str) -> pd.DataFrame:
    if ev_res.empty:
        raise ValueError(
            f"No evaluation results found for test_mode {test_mode}. Please check if the evaluation was run correctly."
        )
    ev_res_subset = ev_res[ev_res["test_mode"] == test_mode]
    eval_results_preds = ev_res_subset[ev_res_subset["rand_setting"] == "predictions"]
    if eval_results_preds.empty:
        raise ValueError(
            f"No evaluation results found for test_mode {test_mode} with predictions. "
            "Please check if the evaluation was run correctly."
        )
    return eval_results_preds


def _draw_critical_difference(
    eval_results_preds: pd.DataFrame,
    test_mode: str,
    custom_id: str,
    result_path: pathlib.Path,
) -> None:
    cd_plot = CriticalDifferencePlot(eval_results_preds=eval_results_preds, metric="MSE")
    cd_plot.draw_and_save(
        out_prefix=Path(result_path) / custom_id / "critical_difference_plots",
        out_suffix=test_mode,
    )


def _draw_violin_and_heatmap_panels(
    eval_results_preds: pd.DataFrame,
    test_mode: str,
    custom_id: str,
    result_path: pathlib.Path,
) -> None:
    for plt_type in ["violinplot", "heatmap"]:
        out_dir = "violin_plots" if plt_type == "violinplot" else "heatmaps"
        for normalized in [False, True]:
            out_suffix = f"algorithms_{test_mode}_normalized" if normalized else f"algorithms_{test_mode}"
            if plt_type == "violinplot":
                out_plot: Violin | Heatmap = Violin(
                    df=eval_results_preds,
                    normalized_metrics=normalized,
                    whole_name=False,
                )
            else:
                out_plot = Heatmap(
                    df=eval_results_preds,
                    normalized_metrics=normalized,
                    whole_name=False,
                )
            out_plot.draw_and_save(
                out_prefix=Path(result_path) / custom_id / out_dir,
                out_suffix=out_suffix,
            )


def _draw_per_group_setting_plots(
    grouping: str,
    ev_res_per_group: pd.DataFrame | None,
    test_mode: str,
    custom_id: str,
    result_path: pathlib.Path,
) -> None:
    if ev_res_per_group is None:
        return
    corr_comp = ComparisonScatter(
        df=ev_res_per_group,
        color_by=grouping,
        test_mode=test_mode,
        algorithm="all",
    )
    if corr_comp.name is not None:
        corr_comp.draw_and_save(
            out_prefix=Path(result_path) / custom_id / "comp_scatter",
            out_suffix=corr_comp.name,
        )


def draw_test_mode_plots(
    test_mode: str,
    ev_res: pd.DataFrame,
    ev_res_per_drug: pd.DataFrame | None,
    ev_res_per_cell_line: pd.DataFrame | None,
    custom_id: str,
    path_data: pathlib.Path,
    result_path: pathlib.Path,
) -> np.ndarray:
    """Draw all plots for one evaluation test mode.

    :param test_mode: Test mode to render (for example ``"LCO"``).
    :param ev_res: Overall evaluation results.
    :param ev_res_per_drug: Per-drug evaluation results.
    :param ev_res_per_cell_line: Per-cell-line evaluation results.
    :param custom_id: Run identifier for output paths.
    :param path_data: Dataset root directory.
    :param result_path: Root results directory.

    :returns: Unique algorithm names in the prediction subset.
    """
    eval_results_preds = _require_prediction_subset(ev_res, test_mode)
    _draw_critical_difference(eval_results_preds, test_mode, custom_id, result_path)
    _draw_violin_and_heatmap_panels(eval_results_preds, test_mode, custom_id, result_path)

    if test_mode in ("LPO", "LDO"):
        _draw_per_group_setting_plots("drug_name", ev_res_per_drug, test_mode, custom_id, result_path)
    if test_mode in ("LPO", "LCO", "LTO"):
        _draw_per_group_setting_plots("cell_line_name", ev_res_per_cell_line, test_mode, custom_id, result_path)

    cross_study_tables = CrossStudyTables(
        evaluation_metrics=ev_res[ev_res["test_mode"] == test_mode], path_data=path_data
    )
    cross_study_tables.draw_and_save(
        out_prefix=Path(result_path) / custom_id / "html_tables",
        out_suffix=test_mode,
    )

    return eval_results_preds["algorithm"].unique()
