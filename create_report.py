import os
import argparse

from drevalpy.visualization import (
    HTMLTable,
    CriticalDifferencePlot,
    Violin,
    Heatmap,
    CorrelationComparisonScatter,
    RegressionSliderPlot,
)

from drevalpy.visualization.utils import (
    parse_results,
    prep_results,
    write_results,
    create_index_html,
    create_html,
)


def create_output_directories(custom_id):
    # if they do not exist yet:
    # make directories: violin_plots, heatmaps, regression_plots, corr_comp_scatter, html_tables, critical_difference_plots
    os.makedirs(f"results/{custom_id}/violin_plots", exist_ok=True)
    os.makedirs(f"results/{custom_id}/heatmaps", exist_ok=True)
    os.makedirs(f"results/{custom_id}/regression_plots", exist_ok=True)
    os.makedirs(f"results/{custom_id}/corr_comp_scatter", exist_ok=True)
    os.makedirs(f"results/{custom_id}/html_tables", exist_ok=True)
    os.makedirs(f"results/{custom_id}/critical_difference_plots", exist_ok=True)


def draw_setting_plots(
    lpo_lco_ldo, ev_res, ev_res_per_drug, ev_res_per_cell_line, custom_id
):
    ev_res_subset = ev_res[ev_res["LPO_LCO_LDO"] == lpo_lco_ldo]
    # PIPELINE: SAVE_TABLES
    html_table = HTMLTable(
        df=ev_res_subset,
        group_by="all",
    )
    html_table.draw_and_save(
        out_prefix=f"results/{custom_id}/html_tables/", out_suffix=lpo_lco_ldo
    )

    # only draw figures for 'real' predictions comparing all models
    eval_results_preds = ev_res_subset[ev_res_subset["rand_setting"] == "predictions"]

    # PIPELINE: DRAW_CRITICAL_DIFFERENCE
    cd_plot = CriticalDifferencePlot(
        eval_results_preds=eval_results_preds, metric="MSE"
    )
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
                out_prefix=f"results/{custom_id}/{out_dir}/", out_suffix=out_suffix
            )

    # per group plots
    if lpo_lco_ldo == "LPO" or lpo_lco_ldo == "LCO":
        draw_per_grouping_setting_plots(
            grouping="drug",
            ev_res_per_group=ev_res_per_drug,
            lpo_lco_ldo=lpo_lco_ldo,
            custom_id=custom_id,
        )
    if lpo_lco_ldo == "LPO" or lpo_lco_ldo == "LDO":
        draw_per_grouping_setting_plots(
            grouping="cell_line",
            ev_res_per_group=ev_res_per_cell_line,
            lpo_lco_ldo=lpo_lco_ldo,
            custom_id=custom_id,
        )

    return eval_results_preds["algorithm"].unique()


def draw_per_grouping_setting_plots(grouping, ev_res_per_group, lpo_lco_ldo, custom_id):
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

    evaluation_results_per_group_subs = ev_res_per_group[
        ev_res_per_group["LPO_LCO_LDO"] == lpo_lco_ldo
    ]
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
    model,
    ev_res,
    ev_res_per_drug,
    ev_res_per_cell_line,
    true_vs_pred,
    lpo_lco_ldo,
    custom_id,
):
    eval_results_algorithm = ev_res[
        (ev_res["LPO_LCO_LDO"] == lpo_lco_ldo) & (ev_res["algorithm"] == model)
    ]
    # PIPELINE: DRAW_VIOLIN_AND_HEATMAP
    for plt_type in ["violinplot", "heatmap"]:
        if plt_type == "violinplot":
            out_dir = "violin_plots"
            out_plot = Violin(
                df=eval_results_algorithm, normalized_metrics=False, whole_name=True
            )
        else:
            out_dir = "heatmaps"
            out_plot = Heatmap(
                df=eval_results_algorithm, normalized_metrics=False, whole_name=True
            )
        out_plot.draw_and_save(
            out_prefix=f"results/{custom_id}/{out_dir}/",
            out_suffix=f"{model}_{lpo_lco_ldo}",
        )

    if lpo_lco_ldo == "LPO" or lpo_lco_ldo == "LCO":
        draw_per_grouping_algorithm_plots(
            grouping_slider="cell_line",
            grouping_scatter_table="drug",
            model=model,
            ev_res_per_group=ev_res_per_drug,
            t_v_p=true_vs_pred,
            lpo_lco_ldo=lpo_lco_ldo,
            custom_id=custom_id,
        )
    if lpo_lco_ldo == "LPO" or lpo_lco_ldo == "LDO":
        draw_per_grouping_algorithm_plots(
            grouping_slider="drug",
            grouping_scatter_table="cell_line",
            model=model,
            ev_res_per_group=ev_res_per_cell_line,
            t_v_p=true_vs_pred,
            lpo_lco_ldo=lpo_lco_ldo,
            custom_id=custom_id,
        )


def draw_per_grouping_algorithm_plots(
    grouping_slider,
    grouping_scatter_table,
    model,
    ev_res_per_group,
    t_v_p,
    lpo_lco_ldo,
    custom_id,
):
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
    parser = argparse.ArgumentParser(
        description="Generate reports from evaluation results"
    )
    parser.add_argument(
        "--run_id", required=True, help="Run ID for the current execution"
    )
    args = parser.parse_args()
    run_id = args.run_id

    # assert that the run_id folder exists
    assert os.path.exists(
        f"results/{run_id}"
    ), f"Folder results/{run_id} does not exist. The pipeline has to be run first."

    # PIPELINE: EVALUATE_FINAL, COLLECT_RESULTS
    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = parse_results(path_to_results=f"results/{run_id}")

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
    evaluation_results = pd.read_csv(f'results/{run_id}/evaluation_results.csv', index_col=0)
    evaluation_results_per_drug = pd.read_csv(f'results/{run_id}/evaluation_results_per_drug.csv', index_col=0)
    evaluation_results_per_cell_line = None
    true_vs_pred = pd.read_csv(f'results/{run_id}/true_vs_pred.csv', index_col=0)
    """

    # create output directories: violin_plots, heatmaps, regression_plots, corr_comp_scatter, html_tables, critical_difference_plot
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
                true_vs_pred=true_vs_pred,
                lpo_lco_ldo=setting,
                custom_id=run_id,
            )
        # get all html files from results/{run_id}
        all_files = []
        for currentpath, folders, files in os.walk(f"results/{run_id}"):
            for file in files:
                if file.endswith(
                        ".html") and file != "index.html" and file != "LPO.html" and file != "LCO.html" and file != "LDO.html":
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
        custom_id=run_id, test_modes=settings, prefix_results=f"results/{run_id}"
    )
