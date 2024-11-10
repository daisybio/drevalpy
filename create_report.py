"""Renders the evaluation results into an HTML report with various plots and tables."""

import argparse
import os

import pandas as pd

from drevalpy.visualization import (
    CorrelationComparisonScatter,
    CriticalDifferencePlot,
    Heatmap,
    HTMLTable,
    RegressionSliderPlot,
    Violin,
)
from drevalpy.visualization.utils import create_html, create_index_html, parse_results, prep_results, write_results


def create_output_directories(custom_id: str) -> None:
    """
    If they do not exist yet, make directories for the visualization files.

    :param custom_id: run id passed via command line
    """
    os.makedirs(f"results/{custom_id}/violin_plots", exist_ok=True)
    os.makedirs(f"results/{custom_id}/heatmaps", exist_ok=True)
    os.makedirs(f"results/{custom_id}/regression_plots", exist_ok=True)
    os.makedirs(f"results/{custom_id}/corr_comp_scatter", exist_ok=True)
    os.makedirs(f"results/{custom_id}/html_tables", exist_ok=True)
    os.makedirs(f"results/{custom_id}/critical_difference_plots", exist_ok=True)


def draw_setting_plots(
    lpo_lco_ldo: str,
    ev_res: pd.DataFrame,
    ev_res_per_drug: pd.DataFrame,
    ev_res_per_cell_line: pd.DataFrame,
    custom_id: str,
) -> list[str]:
    """
    Draw all plots for a specific setting (LPO, LCO, LDO).

    :param lpo_lco_ldo: setting
    :param ev_res: overall evaluation results
    :param ev_res_per_drug: evaluation results per drug
    :param ev_res_per_cell_line: evaluation results per cell line
    :param custom_id: run id passed via command line
    :returns: list of unique algorithms
    """
    ev_res_subset = ev_res[ev_res["LPO_LCO_LDO"] == lpo_lco_ldo]
    # PIPELINE: SAVE_TABLES
    html_table = HTMLTable(
        df=ev_res_subset,
        group_by="all",
    )
    html_table.draw_and_save(out_prefix=f"results/{custom_id}/html_tables/", out_suffix=lpo_lco_ldo)

    # only draw figures for 'real' predictions comparing all models
    eval_results_preds = ev_res_subset[ev_res_subset["rand_setting"] == "predictions"]

    # PIPELINE: DRAW_CRITICAL_DIFFERENCE
    cd_plot = CriticalDifferencePlot(eval_results_preds=eval_results_preds, metric="MSE")
    cd_plot.draw_and_save(
        out_prefix=f"results/{custom_id}/critical_difference_plots/",
        out_suffix=lpo_lco_ldo,
    )

    # PIPELINE: DRAW_VIOLIN_AND_HEATMAP
    for plt_type in ["violinplot", "heatmap"]:
        if plt_type == "violinplot":
            out_dir = "violin_plots"
        else:
            out_dir = "heatmaps"
        for normalized in [False, True]:
            if normalized:
                out_suffix = f"algorithms_{lpo_lco_ldo}_normalized"
            else:
                out_suffix = f"algorithms_{lpo_lco_ldo}"
            if plt_type == "violinplot":
                out_plot = Violin(
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
                out_prefix=f"results/{custom_id}/{out_dir}/",
                out_suffix=out_suffix,
            )

    # per group plots
    if lpo_lco_ldo in ("LPO", "LCO"):
        draw_per_grouping_setting_plots(
            grouping="drug",
            ev_res_per_group=ev_res_per_drug,
            lpo_lco_ldo=lpo_lco_ldo,
            custom_id=custom_id,
        )
    if lpo_lco_ldo in ("LPO", "LDO"):
        draw_per_grouping_setting_plots(
            grouping="cell_line",
            ev_res_per_group=ev_res_per_cell_line,
            lpo_lco_ldo=lpo_lco_ldo,
            custom_id=custom_id,
        )

    return eval_results_preds["algorithm"].unique()


def draw_per_grouping_setting_plots(
    grouping: str, ev_res_per_group: pd.DataFrame, lpo_lco_ldo: str, custom_id: str
) -> None:
    """
    Draw plots for a specific grouping (drug or cell line) for a specific setting (LPO, LCO, LDO).

    :param grouping: drug or cell_line
    :param ev_res_per_group: evaluation results per drug or per cell line
    :param lpo_lco_ldo: setting
    :param custom_id: run id passed over command line
    """
    # PIPELINE: DRAW_CORR_COMP
    corr_comp = CorrelationComparisonScatter(
        df=ev_res_per_group,
        color_by=grouping,
        lpo_lco_ldo=lpo_lco_ldo,
        algorithm="all",
    )
    if corr_comp.name is not None:
        corr_comp.draw_and_save(
            out_prefix=f"results/{custom_id}/corr_comp_scatter/",
            out_suffix=corr_comp.name,
        )

    evaluation_results_per_group_subs = ev_res_per_group[ev_res_per_group["LPO_LCO_LDO"] == lpo_lco_ldo]
    # PIPELINE: SAVE_TABLES
    html_table = HTMLTable(
        df=evaluation_results_per_group_subs,
        group_by=grouping,
    )
    html_table.draw_and_save(
        out_prefix=f"results/{custom_id}/html_tables/",
        out_suffix=f"{grouping}_{lpo_lco_ldo}",
    )


def draw_algorithm_plots(
    model: str,
    ev_res: pd.DataFrame,
    ev_res_per_drug: pd.DataFrame,
    ev_res_per_cell_line: pd.DataFrame,
    t_vs_p: pd.DataFrame,
    lpo_lco_ldo: str,
    custom_id: str,
) -> None:
    """
    Draw all plots for a specific algorithm.

    :param model: name of the model/algorithm
    :param ev_res: overall evaluation results
    :param ev_res_per_drug: evaluation results per drug
    :param ev_res_per_cell_line: evaluation results per cell line
    :param t_vs_p: true response values vs. predicted response values
    :param lpo_lco_ldo: setting
    :param custom_id: run id passed via command line
    """
    eval_results_algorithm = ev_res[(ev_res["LPO_LCO_LDO"] == lpo_lco_ldo) & (ev_res["algorithm"] == model)]
    # PIPELINE: DRAW_VIOLIN_AND_HEATMAP
    for plt_type in ["violinplot", "heatmap"]:
        if plt_type == "violinplot":
            out_dir = "violin_plots"
            out_plot = Violin(
                df=eval_results_algorithm,
                normalized_metrics=False,
                whole_name=True,
            )
        else:
            out_dir = "heatmaps"
            out_plot = Heatmap(
                df=eval_results_algorithm,
                normalized_metrics=False,
                whole_name=True,
            )
        out_plot.draw_and_save(
            out_prefix=f"results/{custom_id}/{out_dir}/",
            out_suffix=f"{model}_{lpo_lco_ldo}",
        )

    if lpo_lco_ldo in ("LPO", "LCO"):
        draw_per_grouping_algorithm_plots(
            grouping_slider="cell_line",
            grouping_scatter_table="drug",
            model=model,
            ev_res_per_group=ev_res_per_drug,
            t_v_p=t_vs_p,
            lpo_lco_ldo=lpo_lco_ldo,
            custom_id=custom_id,
        )
    if lpo_lco_ldo in ("LPO", "LDO"):
        draw_per_grouping_algorithm_plots(
            grouping_slider="drug",
            grouping_scatter_table="cell_line",
            model=model,
            ev_res_per_group=ev_res_per_cell_line,
            t_v_p=t_vs_p,
            lpo_lco_ldo=lpo_lco_ldo,
            custom_id=custom_id,
        )


def draw_per_grouping_algorithm_plots(
    grouping_slider: str,
    grouping_scatter_table: str,
    model: str,
    ev_res_per_group: pd.DataFrame,
    t_v_p: pd.DataFrame,
    lpo_lco_ldo: str,
    custom_id: str,
):
    """
    Draw plots for a specific grouping (drug or cell line) for a specific algorithm.

    :param grouping_slider: the grouping variable for the regression plots
    :param grouping_scatter_table: the grouping variable for the scatter plots.
            If grouping_slider is drug, this should be cell_line and vice versa
    :param model: name of the model/algorithm
    :param ev_res_per_group: evaluation results per drug or per cell line
    :param t_v_p: true response values vs. predicted response values
    :param lpo_lco_ldo: setting
    :param custom_id: run id passed via command line
    """
    # PIPELINE: DRAW_CORR_COMP
    corr_comp = CorrelationComparisonScatter(
        df=ev_res_per_group,
        color_by=grouping_scatter_table,
        lpo_lco_ldo=lpo_lco_ldo,
        algorithm=model,
    )
    if corr_comp.name is not None:
        corr_comp.draw_and_save(
            out_prefix=f"results/{custom_id}/corr_comp_scatter/",
            out_suffix=corr_comp.name,
        )

    # PIPELINE: DRAW_REGRESSION
    for normalize in [False, True]:
        name_suffix = "_normalized" if normalize else ""
        name = f"{lpo_lco_ldo}_{grouping_slider}{name_suffix}"

        regr_slider = RegressionSliderPlot(
            df=t_v_p,
            lpo_lco_ldo=lpo_lco_ldo,
            model=model,
            group_by=grouping_slider,
            normalize=normalize,
        )
        regr_slider.draw_and_save(
            out_prefix=f"results/{custom_id}/regression_plots/",
            out_suffix=f"{name}_{model}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reports from evaluation results")
    parser.add_argument("--run_id", required=True, help="Run ID for the current execution")
    args = parser.parse_args()
    run_id = args.run_id

    # assert that the run_id folder exists
    if not os.path.exists(f"results/{run_id}"):
        raise AssertionError(f"Folder results/{run_id} does not exist. The pipeline has to be run first.")

    # not part of pipeline
    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = parse_results(path_to_results=f"results/{run_id}")

    # part of pipeline: EVALUATE_FINAL, COLLECT_RESULTS
    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = prep_results(
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    )

    write_results(
        path_out=f"results/{run_id}/",
        eval_results=evaluation_results,
        eval_results_per_drug=evaluation_results_per_drug,
        eval_results_per_cl=evaluation_results_per_cell_line,
        t_vs_p=true_vs_pred,
    )
    """
    For debugging:
    evaluation_results = pd.read_csv(
        f'results/{run_id}/evaluation_results.csv', index_col=0
    )
    evaluation_results_per_drug = pd.read_csv(
        f'results/{run_id}/evaluation_results_per_drug.csv', index_col=0
    )
    evaluation_results_per_cell_line = None
    true_vs_pred = pd.read_csv(
        f'results/{run_id}/true_vs_pred.csv', index_col=0
    )
    """

    create_output_directories(run_id)
    # Start loop over all settings
    settings = evaluation_results["LPO_LCO_LDO"].unique()

    for setting in settings:
        print(f"Generating report for {setting} ...")
        unique_algos = draw_setting_plots(
            lpo_lco_ldo=setting,
            ev_res=evaluation_results,
            ev_res_per_drug=evaluation_results_per_drug,
            ev_res_per_cell_line=evaluation_results_per_cell_line,
            custom_id=run_id,
        )
        # draw figures for each algorithm with all randomizations etc
        for algorithm in unique_algos:
            draw_algorithm_plots(
                model=algorithm,
                ev_res=evaluation_results,
                ev_res_per_drug=evaluation_results_per_drug,
                ev_res_per_cell_line=evaluation_results_per_cell_line,
                t_vs_p=true_vs_pred,
                lpo_lco_ldo=setting,
                custom_id=run_id,
            )
        # get all html files from results/{run_id}
        all_files = []
        for _, _, files in os.walk(f"results/{run_id}"):
            for file in files:
                if file.endswith(".html") and file not in ["index.html", "LPO.html", "LCO.html", "LDO.html"]:
                    all_files.append(file)
        # PIPELINE: WRITE_HTML
        create_html(
            run_id=run_id,
            lpo_lco_ldo=setting,
            files=all_files,
            prefix_results=f"results/{run_id}",
        )
    # PIPELINE: WRITE_INDEX
    create_index_html(
        custom_id=run_id,
        test_modes=settings,
        prefix_results=f"results/{run_id}",
    )
