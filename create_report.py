import os
import pandas as pd

from drevalpy.visualization.utils import (
    parse_results,
    parse_layout,
    prep_results,
    write_results,
    draw_violin_or_heatmap,
    draw_scatter_grids_per_group,
    draw_regr_slider,
    export_html_table,
    write_violins_and_heatmaps,
    write_corr_comp_scatter,
)


def create_output_directories(custom_id):
    # if they do not exist yet:
    # make directories: violin_plots, heatmaps, regression_plots, corr_comp_scatter, html_tables
    os.makedirs(f"results/{custom_id}/violin_plots", exist_ok=True)
    os.makedirs(f"results/{custom_id}/heatmaps", exist_ok=True)
    os.makedirs(f"results/{custom_id}/regression_plots", exist_ok=True)
    os.makedirs(f"results/{custom_id}/corr_comp_scatter", exist_ok=True)
    os.makedirs(f"results/{custom_id}/html_tables", exist_ok=True)


def draw_setting_plots(
    lpo_lco_ldo, ev_res, ev_res_per_drug, ev_res_per_cell_line, custom_id
):
    ev_res_subset = ev_res[ev_res["LPO_LCO_LDO"] == lpo_lco_ldo]
    # PIPELINE: SAVE_TABLES
    export_html_table(
        df=ev_res_subset,
        export_path=f"results/{custom_id}/html_tables/table_all_{lpo_lco_ldo}.html",
        grouping="all",
    )

    # only draw figures for 'real' predictions comparing all models
    eval_results_preds = ev_res_subset[ev_res_subset["rand_setting"] == "predictions"]
    # PIPELINE: DRAW_VIOLIN_AND_HEATMAP
    for plt_type in ["violinplot", "heatmap"]:
        if plt_type == "violinplot":
            out_dir = "violin_plots"
        else:
            out_dir = "heatmaps"
        for normalized in [False, True]:
            if normalized:
                outpath = f"results/{custom_id}/{out_dir}/{plt_type}_algorithms_{lpo_lco_ldo}_normalized.html"
            else:
                outpath = f"results/{custom_id}/{out_dir}/{plt_type}_algorithms_{lpo_lco_ldo}.html"
            outplot = draw_violin_or_heatmap(
                plot_type=plt_type,
                df=eval_results_preds,
                normalized_metrics=normalized,
                whole_name=False,
            )
            outplot.fig.write_html(outpath)

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
    draw_scatter_grids_per_group(
        df=ev_res_per_group,
        group_by=grouping,
        lpo_lco_ldo=setting,
        out_prefix=f"results/{custom_id}/corr_comp_scatter/",
        algorithm="all",
    )
    evaluation_results_per_group_subs = ev_res_per_group[
        ev_res_per_group["LPO_LCO_LDO"] == lpo_lco_ldo
    ]
    # PIPELINE: SAVE_TABLES
    export_html_table(
        df=evaluation_results_per_group_subs,
        export_path=f"results/{custom_id}/html_tables/table_{grouping}_{lpo_lco_ldo}.html",
        grouping=grouping,
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
        outplot = draw_violin_or_heatmap(
            plot_type=plt_type,
            df=eval_results_algorithm,
            normalized_metrics=False,
            whole_name=True,
        )
        if plt_type == "violinplot":
            out_dir = "violin_plots"
        else:
            out_dir = "heatmaps"
        outplot.fig.write_html(
            f"results/{custom_id}/{out_dir}/{plt_type}_{model}_{lpo_lco_ldo}.html"
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
    draw_scatter_grids_per_group(
        df=ev_res_per_group,
        group_by=grouping_scatter_table,
        lpo_lco_ldo=lpo_lco_ldo,
        out_prefix=f"results/{custom_id}/corr_comp_scatter/",
        algorithm=model,
    )

    # PIPELINE: DRAW_REGRESSION
    for normalize in [False, True]:
        if normalize:
            name = f"{lpo_lco_ldo}_{grouping_slider}_normalized"
        else:
            name = f"{lpo_lco_ldo}_{grouping_slider}"
        draw_regr_slider(
            t_v_p=t_v_p,
            lpo_lco_ldo=lpo_lco_ldo,
            model=model,
            grouping_slider=grouping_slider,
            out_prefix=f"results/{custom_id}/regression_plots/",
            name=name,
            normalize=normalize,
        )


def create_html(custom_id, setting):
    # copy images to the results directory
    os.system(f"cp drevalpy/visualization/style_utils/favicon.png results/{custom_id}")
    os.system(
        f"cp drevalpy/visualization/style_utils/nf-core-drugresponseeval_logo_light.png results/{custom_id}"
    )
    with open(f"results/{custom_id}/{setting}.html", "w") as f:
        parse_layout(
            f=f, path_to_layout="drevalpy/visualization/style_utils/page_layout.html"
        )
        f.write(f"<h1>Results for {custom_id}: {setting}</h1>\n")

        plot_list = [
            f
            for f in os.listdir(f"results/{custom_id}/violin_plots")
            if setting in f
            and f != f"violinplot_algorithms_{setting}.html"
            and f != f"violinplot_algorithms_{setting}_normalized.html"
        ]
        write_violins_and_heatmaps(
            f=f, setting=setting, plot_list=plot_list, plot="Violin"
        )
        plot_list = [
            f
            for f in os.listdir(f"results/{custom_id}/heatmaps")
            if setting in f
            and f != f"heatmap_algorithms_{setting}.html"
            and f != f"heatmap_algorithms_{setting}_normalized.html"
        ]
        write_violins_and_heatmaps(
            f=f, setting=setting, plot_list=plot_list, plot="Heatmap"
        )

        f.write('<h2 id="regression_plots">Regression plots</h2>\n')
        f.write("<ul>\n")
        file_list = [
            f
            for f in os.listdir(f"results/{custom_id}/regression_plots")
            if setting in f
        ]
        file_list.sort()
        for file in file_list:
            f.write(
                f'<li><a href="regression_plots/{file}" target="_blank">{file}</a></li>\n'
            )
        f.write("</ul>\n")

        f.write('<h2 id="corr_comp">Comparison of correlation metrics</h2>\n')

        group_comparison_list = [
            f
            for f in os.listdir(f"results/{custom_id}/corr_comp_scatter")
            if setting in f and f.split("_")[3] == "drug"
        ]
        write_corr_comp_scatter(
            f=f, setting=setting, group_by="drug", plot_list=group_comparison_list
        )
        group_comparison_list = [
            f
            for f in os.listdir(f"results/{custom_id}/corr_comp_scatter")
            if setting in f and f.split("_")[3] == "cell"
        ]
        write_corr_comp_scatter(
            f=f, setting=setting, group_by="cell_line", plot_list=group_comparison_list
        )

        f.write('<h2 id="tables"> Evaluation Results Table</h2>\n')
        with open(
            f"results/{custom_id}/html_tables/table_all_{setting}.html", "r"
        ) as eval_f:
            eval_results = eval_f.readlines()
            eval_results[0] = eval_results[0].replace(
                '<table border="1" class="dataframe">',
                '<table class="display customDataTable" style="width:100%">',
            )
            for line in eval_results:
                f.write(line)
        if setting != "LCO":
            f.write("<h2> Evaluation Results per Cell Line Table</h2>\n")
            with open(
                f"results/{custom_id}/html_tables/table_cell_line_{setting}.html",
                "r",
            ) as eval_f:
                eval_results = eval_f.readlines()
                eval_results[0] = eval_results[0].replace(
                    '<table border="1" class="dataframe">',
                    '<table class="display customDataTable" style="width:100%">',
                )
                for line in eval_results:
                    f.write(line)
        if setting != "LDO":
            f.write("<h2> Evaluation Results per Drug Table</h2>\n")
            with open(
                f"results/{custom_id}/html_tables/table_drug_{setting}.html", "r"
            ) as eval_f:
                eval_results = eval_f.readlines()
                eval_results[0] = eval_results[0].replace(
                    '<table border="1" class="dataframe">',
                    '<table class="display customDataTable" style="width:100%">',
                )
                for line in eval_results:
                    f.write(line)
        f.write("</div>\n")
        f.write("</body>\n")
        f.write("</html>\n")


def create_index_html(custom_id):
    # copy images to the results directory
    os.system(f"cp drevalpy/visualization/style_utils/LPO.png results/{custom_id}")
    os.system(f"cp drevalpy/visualization/style_utils/LCO.png results/{custom_id}")
    os.system(f"cp drevalpy/visualization/style_utils/LDO.png results/{custom_id}")
    with open(f"results/{custom_id}/index.html", "w") as f:
        parse_layout(
            f=f, path_to_layout="drevalpy/visualization/style_utils/index_layout.html"
        )
        f.write('<div class="main">\n')
        f.write(
            '<img src="nf-core-drugresponseeval_logo_light.png" width="364px" height="100px" alt="Logo">\n'
        )
        f.write(f"<h1>Results for {custom_id}</h1>\n")
        f.write("<h2>Available settings</h2>\n")
        f.write('<div style="display: inline-block;">\n')
        f.write(
            "<p>Click on the images to open the respective report in a new tab.</p>\n"
        )
        settings = [
            f.split(".html")[0]
            for f in os.listdir(f"results/{custom_id}")
            if f.endswith(".html") and f.startswith("L")
        ]
        settings.sort()
        for setting in settings:
            f.write(
                f'<a href="{setting}.html" target="_blank"><img src="{setting}.png" style="width:300px;height:300px;"></a>\n'
            )
        f.write("</div>\n")
        f.write("</div>\n")
        f.write("</body>\n")
        f.write("</html>\n")


if __name__ == "__main__":
    # Load the dataset
    run_id = "myRun"

    # PIPELINE: COLLECT_RESULTS
    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = parse_results(path_to_results=f"results/{run_id}", path_out=f"results/{run_id}")

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

    write_results(path_out=f"results/{run_id}/",
                  eval_results=evaluation_results,
                  eval_results_per_drug=evaluation_results_per_drug,
                  eval_results_per_cl=evaluation_results_per_cell_line,
                  t_vs_p=true_vs_pred)
    '''
    evaluation_results = pd.read_csv(f'results/{run_id}/evaluation_results.csv', index_col=0)
    evaluation_results_per_drug = pd.read_csv(f'results/{run_id}/evaluation_results_per_drug.csv', index_col=0)
    evaluation_results_per_cell_line = pd.read_csv(f'results/{run_id}/evaluation_results_per_cl.csv',
                                             index_col=0)
    true_vs_pred = pd.read_csv(f'results/{run_id}/true_vs_pred.csv', index_col=0)
    '''

    # create output directories: violin_plots, heatmaps, regression_plots, corr_comp_scatter, html_tables
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
        # write individual html
        create_html(run_id, setting)
    create_index_html(run_id)
