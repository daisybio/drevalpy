import os
import pandas as pd

from utils import (
    parse_layout,
    prep_results,
    write_violins_and_heatmaps,
    write_scatter_eval_models,
)
from heatmap import Heatmap
from regression_slider_plot import RegressionSliderPlot
from violin import Violin
from corr_comp_scatter import CorrelationComparisonScatter


def create_html(run_id, setting):
    # copy images to the results directory
    os.system(f"cp style_utils/favicon.png ../results/{run_id}")
    os.system(
        f"cp style_utils/nf-core-drugresponseeval_logo_light.png ../results/{run_id}"
    )
    with open(f"../results/{run_id}/{setting}.html", "w") as f:
        parse_layout(f=f, path_to_layout="style_utils/page_layout.html")
        f.write(f"<h1>Results for {run_id}: {setting}</h1>\n")

        plot_list = [
            f
            for f in os.listdir(f"../results/{run_id}/violin_plots")
            if setting in f
            and f != f"violinplot_{setting}.html"
            and f != f"violinplot_{setting}_normalized.html"
        ]
        write_violins_and_heatmaps(
            f=f, setting=setting, plot_list=plot_list, plot="Violin"
        )
        plot_list = [
            f
            for f in os.listdir(f"../results/{run_id}/heatmaps")
            if setting in f
            and f != f"heatmap_{setting}.html"
            and f != f"heatmap_{setting}_normalized.html"
        ]
        write_violins_and_heatmaps(
            f=f, setting=setting, plot_list=plot_list, plot="Heatmap"
        )

        f.write('<h2 id="regression_plots">Regression plots</h2>\n')
        f.write("<ul>\n")
        file_list = [
            f
            for f in os.listdir(f"../results/{run_id}/regression_plots")
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
            for f in os.listdir(f"../results/{run_id}/corr_comp_scatter")
            if setting in f and f.endswith("drug.html")
        ]
        write_scatter_eval_models(
            f=f, setting=setting, group_by="drug", plot_list=group_comparison_list
        )
        group_comparison_list = [
            f
            for f in os.listdir(f"../results/{run_id}/corr_comp_scatter")
            if setting in f and f.endswith("cell_line.html")
        ]
        write_scatter_eval_models(
            f=f, setting=setting, group_by="cell_line", plot_list=group_comparison_list
        )

        f.write('<h2 id="tables"> Evaluation Results Table</h2>\n')
        with open(
            f"../results/{run_id}/evaluation_results_{setting}.html", "r"
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
                f"../results/{run_id}/evaluation_results_per_cell_line_{setting}.html",
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
                f"../results/{run_id}/evaluation_results_per_drug_{setting}.html", "r"
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


def create_index_html(run_id):
    # copy images to the results directory
    os.system(f"cp style_utils/LPO.png ../results/{run_id}")
    os.system(f"cp style_utils/LCO.png ../results/{run_id}")
    os.system(f"cp style_utils/LDO.png ../results/{run_id}")
    with open(f"../results/{run_id}/index.html", "w") as f:
        parse_layout(f=f, path_to_layout="style_utils/index_layout.html")
        f.write('<div class="main">\n')
        f.write(
            '<img src="nf-core-drugresponseeval_logo_light.png" width="364px" height="100px" alt="Logo">\n'
        )
        f.write(f"<h1>Results for {run_id}</h1>\n")
        f.write("<h2>Available settings</h2>\n")
        f.write('<div style="display: inline-block;">\n')
        f.write(
            "<p>Click on the images to open the respective report in a new tab.</p>\n"
        )
        settings = [
            f.split(".html")[0]
            for f in os.listdir(f"../results/{run_id}")
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


def draw_violin_and_heatmap(
    df, run_id, plotname, whole_name=False, normalized_metrics=False
):
    if not os.path.exists(f"../results/{run_id}/violin_plots"):
        os.mkdir(f"../results/{run_id}/violin_plots")
    if not os.path.exists(f"../results/{run_id}/heatmaps"):
        os.mkdir(f"../results/{run_id}/heatmaps")
    violin = Violin(df, normalized_metrics=normalized_metrics, whole_name=whole_name)
    violin.fig.write_html(
        f"../results/{run_id}/violin_plots/violinplot_{plotname}.html"
    )
    heatmap = Heatmap(df, normalized_metrics=normalized_metrics, whole_name=whole_name)
    heatmap.fig.write_html(f"../results/{run_id}/heatmaps/heatmap_{plotname}.html")


def draw_scatter_grids_per_group(eval_res_group, group_by, setting, run_id):
    if not os.path.exists(f"../results/{run_id}/scatter_eval_models"):
        os.mkdir(f"../results/{run_id}/scatter_eval_models")
    eval_res_group_subset = eval_res_group[eval_res_group["LPO_LCO_LDO"] == setting]
    eval_res_group_models = eval_res_group_subset[
        eval_res_group_subset["rand_setting"] == "predictions"
    ]
    # draw plots for comparison between all models
    corr_comp_scatter = CorrelationComparisonScatter(
        eval_res_group_models, color_by=group_by
    )
    corr_comp_scatter.dropdown_fig.write_html(
        f"../results/{run_id}/scatter_eval_models/scatter_eval_models_{group_by}_{setting}.html"
    )
    corr_comp_scatter.fig_overall.write_html(
        f"../results/{run_id}/scatter_eval_models/scatter_eval_models_{group_by}_overall_{setting}.html"
    )

    # draw plots per model: compare between original model and models with modification
    for algorithm in eval_res_group_models["algorithm"].unique():
        eval_res_group_algorithm = eval_res_group_subset[
            eval_res_group_subset["algorithm"] == algorithm
        ]
        corr_comp_scatter = CorrelationComparisonScatter(
            eval_res_group_algorithm, color_by=group_by
        )
        corr_comp_scatter.dropdown_fig.write_html(
            f"../results/{run_id}/scatter_eval_models/scatter_eval_models_{group_by}_{algorithm}_{setting}.html"
        )
        corr_comp_scatter.fig_overall.write_html(
            f"../results/{run_id}/scatter_eval_models/scatter_eval_models_{group_by}_overall_{algorithm}_{setting}.html"
        )


if __name__ == "__main__":
    # Load the dataset
    run_id = "test3"
    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = prep_results(path_to_results=f"../results/{run_id}")
    if not os.path.exists(f"../results/{run_id}/regression_plots"):
        os.mkdir(f"../results/{run_id}/regression_plots")

    settings = evaluation_results["LPO_LCO_LDO"].unique()

    for setting in settings:
        print(f"Generating report for {setting} ...")
        eval_results_subset = evaluation_results[
            evaluation_results["LPO_LCO_LDO"] == setting
        ]
        true_vs_pred_subset = true_vs_pred[true_vs_pred["LPO_LCO_LDO"] == setting]

        # only draw figures for 'real' predictions comparing all models
        eval_results_algorithms = eval_results_subset[
            eval_results_subset["rand_setting"] == "predictions"
        ]
        draw_violin_and_heatmap(
            eval_results_algorithms, run_id, f"algorithms_{setting}"
        )

        # draw the same figures but with drug/cell-line normalized metrics
        draw_violin_and_heatmap(
            eval_results_algorithms,
            run_id,
            f"algorithms_{setting}_normalized",
            normalized_metrics=True,
        )

        # draw figures for each algorithm with all randomizations etc
        for algorithm in eval_results_algorithms["algorithm"].unique():
            eval_results_algorithm = eval_results_subset[
                eval_results_subset["algorithm"] == algorithm
            ]
            draw_violin_and_heatmap(
                eval_results_algorithm,
                run_id,
                f"{algorithm}_{setting}",
                whole_name=True,
            )

        if setting == "LPO" or setting == "LCO":
            # draw correlation comparison scatter plots (overall figure & drop down plot)
            draw_scatter_grids_per_group(
                eval_res_group=evaluation_results_per_drug,
                group_by="drug",
                setting=setting,
                run_id=run_id,
            )
            # export table to html
            evaluation_results_per_drug_subs = evaluation_results_per_drug[
                evaluation_results_per_drug["LPO_LCO_LDO"] == setting
            ]
            evaluation_results_per_drug_subs = evaluation_results_per_drug_subs[
                [
                    "drug",
                    "algorithm",
                    "rand_setting",
                    "CV_split",
                    "MSE",
                    "R^2",
                    "Pearson",
                    "RMSE",
                    "MAE",
                    "Spearman",
                    "Kendall",
                    "Partial_Correlation",
                    "LPO_LCO_LDO",
                ]
            ]
            evaluation_results_per_drug_subs.to_html(
                f"../results/{run_id}/evaluation_results_per_drug_{setting}.html",
                index=False,
            )
            for algorithm in true_vs_pred_subset["algorithm"].unique():
                t_vs_pred_algo = true_vs_pred_subset[
                    true_vs_pred_subset["algorithm"] == algorithm
                ]
                # generate regression plots
                regr_slider = RegressionSliderPlot(
                    df=t_vs_pred_algo, group_by="cell_line"
                )
                regr_slider.fig.write_html(
                    f"../results/{run_id}/regression_plots/{setting}_{algorithm}_regression_lines_cell_line.html"
                )
                regr_slider_norm = RegressionSliderPlot(
                    df=t_vs_pred_algo,
                    group_by="cell_line",
                    normalize=True,
                )
                regr_slider_norm.fig.write_html(
                    f"../results/{run_id}/regression_plots/{setting}_{algorithm}_regression_lines_cell_line_normalized.html"
                )

        if setting == "LPO" or setting == "LDO":
            # draw correlation comparison scatter plots (overall figure & drop down plot)
            draw_scatter_grids_per_group(
                eval_res_group=evaluation_results_per_cell_line,
                group_by="cell_line",
                setting=setting,
                run_id=run_id,
            )
            # export table to html
            evaluation_results_per_cell_line_subs = evaluation_results_per_cell_line[
                evaluation_results_per_cell_line["LPO_LCO_LDO"] == setting
            ]
            evaluation_results_per_cell_line_subs = (
                evaluation_results_per_cell_line_subs[
                    [
                        "cell_line",
                        "algorithm",
                        "rand_setting",
                        "CV_split",
                        "MSE",
                        "R^2",
                        "Pearson",
                        "RMSE",
                        "MAE",
                        "Spearman",
                        "Kendall",
                        "Partial_Correlation",
                        "LPO_LCO_LDO",
                    ]
                ]
            )
            evaluation_results_per_cell_line_subs.to_html(
                f"../results/{run_id}/evaluation_results_per_cell_line_{setting}.html",
                index=False,
            )
            for algorithm in true_vs_pred_subset["algorithm"].unique():
                t_vs_pred_algo = true_vs_pred_subset[
                    true_vs_pred_subset["algorithm"] == algorithm
                ]
                # generate regression plots
                regr_slider = RegressionSliderPlot(df=t_vs_pred_algo, group_by="drug")
                regr_slider.fig.write_html(
                    f"../results/{run_id}/regression_plots/{setting}_{algorithm}_regression_lines_drug.html"
                )
                regr_slider_norm = RegressionSliderPlot(
                    df=t_vs_pred_algo, group_by="drug", normalize=True
                )
                regr_slider_norm.fig.write_html(
                    f"../results/{run_id}/regression_plots/{setting}_{algorithm}_regression_lines_drug_normalized.html"
                )
        # reorder columns, export table as html
        eval_results_subset = eval_results_subset[
            [
                "algorithm",
                "rand_setting",
                "CV_split",
                "MSE",
                "R^2",
                "Pearson",
                "R^2: drug normalized",
                "Pearson: drug normalized",
                "R^2: cell_line normalized",
                "Pearson: cell_line normalized",
                "RMSE",
                "MAE",
                "Spearman",
                "Kendall",
                "Partial_Correlation",
                "Spearman: drug normalized",
                "Kendall: drug normalized",
                "Partial_Correlation: drug normalized",
                "Spearman: cell_line normalized",
                "Kendall: cell_line normalized",
                "Partial_Correlation: cell_line normalized",
                "LPO_LCO_LDO",
            ]
        ]
        eval_results_subset.to_html(
            f"../results/{run_id}/evaluation_results_{setting}.html", index=False
        )
        # write individual html
        create_html(run_id, setting)
    create_index_html(run_id)
