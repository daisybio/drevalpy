import shutil
from typing import Dict, List

import importlib_resources
import pandas as pd
import pathlib
import os

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.evaluation import evaluate, AVAILABLE_METRICS
from drevalpy.visualization.violin import Violin
from drevalpy.visualization.heatmap import Heatmap
from drevalpy.visualization.corr_comp_scatter import CorrelationComparisonScatter
from drevalpy.visualization.regression_slider_plot import RegressionSliderPlot
from drevalpy.visualization.critical_difference_plot import CriticalDifferencePlot


def parse_layout(f, path_to_layout):
    with open(path_to_layout, "r") as layout_f:
        layout = layout_f.readlines()
    if path_to_layout.endswith("index_layout.html"):
        # remove the last 2 lines (</body>, </html>)
        layout = layout[:-2]
    else:
        # remove the last 3 lines (</div>, </body>, </html>)
        layout = layout[:-3]
    f.write("".join(layout))


def parse_results(path_to_results: str):
    print("Generating result tables ...")
    # generate list of all result files
    result_dir = pathlib.Path(path_to_results)
    result_files = list(result_dir.rglob("*.csv"))
    result_files = [
        file
        for file in result_files
        if file.name
        not in [
            "evaluation_results.csv",
            "evaluation_results_per_drug.csv",
            "evaluation_results_per_cl.csv",
            "true_vs_pred.csv",
        ]
        and "cv_split" not in file.name
    ]

    # inititalize dictionaries to store the evaluation results
    evaluation_results = None
    evaluation_results_per_drug = None
    evaluation_results_per_cell_line = None
    true_vs_pred = None

    # read every result file and compute the evaluation metrics
    for file in result_files:
        file_parts = os.path.normpath(file).split("/")
        lpo_lco_ldo = file_parts[-4]
        algorithm = file_parts[-3]
        (
            overall_eval,
            eval_results_per_drug,
            eval_results_per_cl,
            t_vs_p,
            model_name,
        ) = evaluate_file(pred_file=file, test_mode=lpo_lco_ldo, model_name=algorithm)

        evaluation_results = (
            overall_eval
            if evaluation_results is None
            else pd.concat([evaluation_results, overall_eval])
        )
        true_vs_pred = (
            t_vs_p if true_vs_pred is None else pd.concat([true_vs_pred, t_vs_p])
        )

        if eval_results_per_drug is not None:
            evaluation_results_per_drug = (
                eval_results_per_drug
                if evaluation_results_per_drug is None
                else pd.concat([evaluation_results_per_drug, eval_results_per_drug])
            )

        if eval_results_per_cl is not None:
            evaluation_results_per_cell_line = (
                eval_results_per_cl
                if evaluation_results_per_cell_line is None
                else pd.concat([evaluation_results_per_cell_line, eval_results_per_cl])
            )

    return (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    )


def evaluate_file(pred_file: pathlib.Path, test_mode: str, model_name: str):
    print("Parsing file:", os.path.normpath(pred_file))
    result = pd.read_csv(pred_file)
    dataset = DrugResponseDataset(
        response=result["response"],
        cell_line_ids=result["cell_line_ids"],
        drug_ids=result["drug_ids"],
        predictions=result["predictions"],
    )
    model = generate_model_names(
        test_mode=test_mode, model_name=model_name, pred_file=pred_file
    )

    # overall evaluation
    overall_eval = {model: evaluate(dataset, AVAILABLE_METRICS.keys())}

    true_vs_pred = pd.DataFrame(
        {
            "model": [model for _ in range(len(dataset.response))],
            "drug": dataset.drug_ids,
            "cell_line": dataset.cell_line_ids,
            "y_true": dataset.response,
            "y_pred": dataset.predictions,
        }
    )

    evaluation_results_per_drug = None
    evaluation_results_per_cl = None
    norm_drug_eval_results = dict()
    norm_cl_eval_results = dict()

    if "LPO" in model or "LCO" in model:
        norm_drug_eval_results, evaluation_results_per_drug = evaluate_per_group(
            df=true_vs_pred,
            group_by="drug",
            norm_group_eval_results=norm_drug_eval_results,
            eval_results_per_group=evaluation_results_per_drug,
            model=model,
        )
    if "LPO" in model or "LDO" in model:
        norm_cl_eval_results, evaluation_results_per_cl = evaluate_per_group(
            df=true_vs_pred,
            group_by="cell_line",
            norm_group_eval_results=norm_cl_eval_results,
            eval_results_per_group=evaluation_results_per_cl,
            model=model,
        )
    overall_eval = pd.DataFrame.from_dict(overall_eval, orient="index")
    if len(norm_drug_eval_results) > 0:
        overall_eval = concat_results(norm_drug_eval_results, "drug", overall_eval)
    if len(norm_cl_eval_results) > 0:
        overall_eval = concat_results(norm_cl_eval_results, "cell_line", overall_eval)

    return (
        overall_eval,
        evaluation_results_per_drug,
        evaluation_results_per_cl,
        true_vs_pred,
        model,
    )


def concat_results(norm_group_res, group_by, eval_res):
    norm_group_res = pd.DataFrame.from_dict(norm_group_res, orient="index")
    # append 'group normalized ' to the column names
    norm_group_res.columns = [
        f"{col}: {group_by} normalized" for col in norm_group_res.columns
    ]
    eval_res = pd.concat([eval_res, norm_group_res], axis=1)
    return eval_res


def prep_results(
    eval_results, eval_results_per_drug, eval_results_per_cell_line, t_vs_p
):
    # add variables
    # split the index by "_" into: algorithm, randomization, setting, split, CV_split
    new_columns = eval_results.index.str.split("_", expand=True).to_frame()
    new_columns.columns = [
        "algorithm",
        "rand_setting",
        "LPO_LCO_LDO",
        "split",
        "CV_split",
    ]
    new_columns.index = eval_results.index
    eval_results = pd.concat([new_columns.drop("split", axis=1), eval_results], axis=1)
    eval_results_per_drug[
        ["algorithm", "rand_setting", "LPO_LCO_LDO", "split", "CV_split"]
    ] = eval_results_per_drug["model"].str.split("_", expand=True)
    eval_results_per_cell_line[
        ["algorithm", "rand_setting", "LPO_LCO_LDO", "split", "CV_split"]
    ] = eval_results_per_cell_line["model"].str.split("_", expand=True)
    t_vs_p[["algorithm", "rand_setting", "LPO_LCO_LDO", "split", "CV_split"]] = t_vs_p[
        "model"
    ].str.split("_", expand=True)

    return eval_results, eval_results_per_drug, eval_results_per_cell_line, t_vs_p


def generate_model_names(test_mode, model_name, pred_file):
    file_parts = os.path.basename(pred_file).split("_")
    pred_rand_rob = file_parts[0]
    if pred_rand_rob == "predictions":
        pred_setting = "predictions"
    elif pred_rand_rob == "randomization":
        pred_setting = "randomize-" + "-".join(file_parts[1:-2])
    elif pred_rand_rob == "robustness_test":
        pred_setting = "-".join(file_parts[:2])
    else:
        raise ValueError(f"Unknown prediction setting: {pred_rand_rob}")
    split = "_".join(os.path.basename(pred_file).split(".")[0].split("_")[-2:])
    return f"{model_name}_{pred_setting}_{test_mode}_{split}"


def evaluate_per_group(
    df, group_by, norm_group_eval_results, eval_results_per_group, model
):
    # calculate the mean of y_true per drug
    print(f"Calculating {group_by}-wise evaluation measures …")
    df[f"mean_y_true_per_{group_by}"] = df.groupby(group_by)["y_true"].transform("mean")
    norm_df = df.copy()
    norm_df["y_true"] = norm_df["y_true"] - norm_df[f"mean_y_true_per_{group_by}"]
    norm_df["y_pred"] = norm_df["y_pred"] - norm_df[f"mean_y_true_per_{group_by}"]
    norm_group_eval_results[model] = evaluate(
        DrugResponseDataset(
            response=norm_df["y_true"],
            cell_line_ids=norm_df["cell_line"],
            drug_ids=norm_df["drug"],
            predictions=norm_df["y_pred"],
        ),
        AVAILABLE_METRICS.keys() - {"MSE", "RMSE", "MAE"},
    )
    # evaluation per group
    eval_results_per_group = compute_evaluation(
        df, eval_results_per_group, group_by, model
    )
    return norm_group_eval_results, eval_results_per_group


def compute_evaluation(df, return_df, group_by, model):
    result_per_group = df.groupby(group_by).apply(
        lambda x: evaluate(
            DrugResponseDataset(
                response=x["y_true"],
                cell_line_ids=x["cell_line"],
                drug_ids=x["drug"],
                predictions=x["y_pred"],
            ),
            AVAILABLE_METRICS.keys(),
        )
    )
    groups = result_per_group.index
    result_per_group = pd.json_normalize(result_per_group)
    result_per_group[group_by] = groups
    result_per_group["model"] = model
    if return_df is None:
        return_df = pd.DataFrame(result_per_group)
    else:
        return_df = pd.concat([return_df, result_per_group])
    return return_df


def draw_critical_difference_plot(
    evaluation_results: pd.DataFrame, path_out: str, metric: str
) -> None:
    out = CriticalDifferencePlot(eval_results_preds=evaluation_results, metric=metric)
    out.fig.savefig(path_out, bbox_inches="tight")


def draw_violin_or_heatmap(plot_type, df, normalized_metrics, whole_name):
    if plot_type == "violinplot":
        out = Violin(
            df=df, normalized_metrics=normalized_metrics, whole_name=whole_name
        )
    else:
        out = Heatmap(
            df=df, normalized_metrics=normalized_metrics, whole_name=whole_name
        )
    return out


def draw_scatter_grids_per_group(df, group_by, lpo_lco_ldo, out_prefix, algorithm=None):
    if group_by == "drug":
        exclude_models = {"NaiveDrugMeanPredictor"}
    else:
        exclude_models = {"NaiveCellLineMeanPredictor"}
    exclude_models = exclude_models.union({"NaivePredictor"})
    if algorithm == "all":
        # draw plots for comparison between all models
        df = df[
            (df["LPO_LCO_LDO"] == lpo_lco_ldo)
            & (df["rand_setting"] == "predictions")
            & (~df["algorithm"].isin(exclude_models))
        ]
        corr_comp_scatter = CorrelationComparisonScatter(df=df, color_by=group_by)
        name = f"{group_by}_{lpo_lco_ldo}"
    elif algorithm not in exclude_models:
        # draw plots for comparison between all test settings of one model
        df = df[(df["LPO_LCO_LDO"] == lpo_lco_ldo) & (df["algorithm"] == algorithm)]
        corr_comp_scatter = CorrelationComparisonScatter(df=df, color_by=group_by)
        name = f"{group_by}_{algorithm}_{lpo_lco_ldo}"
    else:
        return
    corr_comp_scatter.dropdown_fig.write_html(
        f"{out_prefix}corr_comp_scatter_{name}.html"
    )
    corr_comp_scatter.fig_overall.write_html(
        f"{out_prefix}corr_comp_scatter_overall_{name}.html"
    )


def draw_regr_slider(
    t_v_p, lpo_lco_ldo, model, grouping_slider, out_prefix, name, normalize
):
    t_vs_pred_model = t_v_p[
        (t_v_p["LPO_LCO_LDO"] == lpo_lco_ldo) & (t_v_p["algorithm"] == model)
    ]

    regr_slider = RegressionSliderPlot(
        df=t_vs_pred_model, group_by=grouping_slider, normalize=normalize
    )

    out_path = f"{out_prefix}regression_lines_{name}_{model}.html"
    regr_slider.fig.write_html(out_path)


def export_html_table(df, export_path, grouping):
    selected_columns = [
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
    if grouping == "drug":
        selected_columns = ["drug"] + selected_columns
    elif grouping == "cell_line":
        selected_columns = ["cell_line"] + selected_columns
    else:
        selected_columns = [
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
    # reorder columns, export table as html
    df = df[selected_columns]
    df.to_html(export_path, index=False)


def write_results(
    path_out, eval_results, eval_results_per_drug, eval_results_per_cl, t_vs_p
):
    eval_results.to_csv(f"{path_out}evaluation_results.csv", index=True)
    eval_results_per_drug.to_csv(
        f"{path_out}evaluation_results_per_drug.csv", index=True
    )
    eval_results_per_cl.to_csv(f"{path_out}evaluation_results_per_cl.csv", index=True)
    t_vs_p.to_csv(f"{path_out}true_vs_pred.csv", index=True)


def write_violins_and_heatmaps(f, setting, plot_list, plot="Violin"):
    if plot == "Violin":
        nav_id = "violin"
        dir_name = "violin_plots"
        prefix = "violinplot"
    else:
        nav_id = "heatmap"
        dir_name = "heatmaps"
        prefix = "heatmap"
    f.write(
        f'<h2 id="{nav_id}">{plot} Plots of Performance Measures over CV runs</h2>\n'
    )
    f.write(f"<h3>{plot} plots comparing all models</h3>\n")
    f.write(
        f'<iframe src="{dir_name}/{prefix}_algorithms_{setting}.html" width="100%" height="100%" frameBorder="0"></iframe>\n'
    )
    f.write(f"<h3>{plot} plots comparing all models with normalized metrics</h3>\n")
    f.write(
        f"Before calculating the evaluation metrics, all values were normalized by the mean of the drug or cell line. "
        f"Since this only influences the R^2 and the correlation metrics, the error metrics are not shown. \n"
    )
    f.write(
        f'<iframe src="{dir_name}/{prefix}_algorithms_{setting}_normalized.html" width="100%" height="100%" frameBorder="0"></iframe>\n'
    )
    f.write(
        f"<h3>{plot} plots comparing performance measures for tests within each model</h3>\n"
    )
    f.write("<ul>")
    for plot in plot_list:
        f.write(f'<li><a href="{dir_name}/{plot}" target="_blank">{plot}</a></li>\n')
    f.write("</ul>\n")


def write_corr_comp_scatter(f, setting, group_by, plot_list):
    if len(plot_list) > 0:
        f.write(
            f'<h3 id="corr_comp_drug">{group_by.capitalize()}-wise comparison</h3>\n'
        )
        f.write("<h4>Overall comparison between models</h4>\n")
        f.write(
            f'<iframe src="corr_comp_scatter/corr_comp_scatter_overall_{group_by}_{setting}.html" width="100%" height="100%" frameBorder="0"></iframe>\n'
        )
        f.write("<h4>Comparison between all models, dropdown menu</h4>\n")
        f.write(
            f'<iframe src="corr_comp_scatter/corr_comp_scatter_{group_by}_{setting}.html" width="100%" height="100%" frameBorder="0"></iframe>\n'
        )
        f.write("<h4>Comparisons per model</h4>\n")
        f.write("<ul>\n")
        plot_list = [
            elem
            for elem in plot_list
            if elem != f"corr_comp_scatter_{setting}_{group_by}.html"
            and elem != f"corr_comp_scatter_overall_{setting}_{group_by}.html"
        ]
        plot_list.sort()
        for group_comparison in plot_list:
            f.write(
                f'<li><a href="corr_comp_scatter/{group_comparison}" target="_blank">{group_comparison}</a></li>\n'
            )
        f.write("</ul>\n")


def create_index_html(custom_id: str, test_modes: List[str], prefix_results: str):
    # copy images to the results directory
    file_to_copy = [
        "favicon.png",
        "nf-core-drugresponseeval_logo_light.png",
    ]
    for file in file_to_copy:
        file_path = os.path.join(
            str(importlib_resources.files("drevalpy")), "visualization", "style_utils", file
        )
        shutil.copyfile(file_path, os.path.join(prefix_results, file))

    layout_path = os.path.join(str(importlib_resources.files("drevalpy")), "visualization", "style_utils",
                               "index_layout.html")
    idx_html_path = os.path.join(prefix_results, "index.html")
    with open(idx_html_path, "w") as f:
        parse_layout(f=f, path_to_layout=layout_path)
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

        test_modes.sort()
        for lpo_lco_ldo in test_modes:
            img_path = os.path.join(
                str(importlib_resources.files("drevalpy")), "visualization", "style_utils", f"{lpo_lco_ldo}.png"
            )
            shutil.copyfile(img_path, os.path.join(prefix_results, f"{lpo_lco_ldo}.png"))
            f.write(
                f'<a href="{lpo_lco_ldo}.html" target="_blank"><img src="{lpo_lco_ldo}.png" style="width:300px;height:300px;"></a>\n'
            )
        f.write("</div>\n")
        f.write("</div>\n")
        f.write("</body>\n")
        f.write("</html>\n")
