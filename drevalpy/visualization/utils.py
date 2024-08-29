"""
Utility functions for the visualization part of the package.
"""

import os
import pathlib
import shutil
from typing import List
import importlib_resources
import pandas as pd


from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.evaluation import evaluate, AVAILABLE_METRICS
from drevalpy.visualization import HTMLTable
from drevalpy.visualization.vioheat import VioHeat
from drevalpy.visualization.corr_comp_scatter import CorrelationComparisonScatter
from drevalpy.visualization.regression_slider_plot import RegressionSliderPlot
from drevalpy.visualization.critical_difference_plot import CriticalDifferencePlot


def parse_layout(f, path_to_layout):
    """
    Parse the layout file and write it to the output file.
    :param f:
    :param path_to_layout:
    :return:
    """
    with open(path_to_layout, "r", encoding="utf-8") as layout_f:
        layout = layout_f.readlines()
    if path_to_layout.endswith("index_layout.html"):
        # remove the last 2 lines (</body>, </html>)
        layout = layout[:-2]
    else:
        # remove the last 3 lines (</div>, </body>, </html>)
        layout = layout[:-3]
    f.write("".join(layout))


def parse_results(path_to_results: str):
    """
    Parse the results from the given directory.
    :param path_to_results:
    :return:
    """
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
    """
    Evaluate the predictions from the final models.
    :param pred_file:
    :param test_mode:
    :param model_name:
    :return:
    """
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
    norm_drug_eval_results = {}
    norm_cl_eval_results = {}

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
    """
    Concatenate the normalized group results to the evaluation results.
    :param norm_group_res:
    :param group_by:
    :param eval_res:
    :return:
    """
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
    """
    Prepare the results by introducing new columns for algorithm, randomization, setting, split,
    CV_split.
    :param eval_results:
    :param eval_results_per_drug:
    :param eval_results_per_cell_line:
    :param t_vs_p:
    :return:
    """
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
    if eval_results_per_drug is not None:
        eval_results_per_drug[
            ["algorithm", "rand_setting", "LPO_LCO_LDO", "split", "CV_split"]
        ] = eval_results_per_drug["model"].str.split("_", expand=True)
    if eval_results_per_cell_line is not None:
        eval_results_per_cell_line[
            ["algorithm", "rand_setting", "LPO_LCO_LDO", "split", "CV_split"]
        ] = eval_results_per_cell_line["model"].str.split("_", expand=True)
    t_vs_p[["algorithm", "rand_setting", "LPO_LCO_LDO", "split", "CV_split"]] = t_vs_p[
        "model"
    ].str.split("_", expand=True)

    return eval_results, eval_results_per_drug, eval_results_per_cell_line, t_vs_p


def generate_model_names(test_mode, model_name, pred_file):
    """
    Generate the model names based on the prediction file.
    :param test_mode:
    :param model_name:
    :param pred_file:
    :return:
    """
    file_parts = os.path.basename(pred_file).split("_")
    pred_rand_rob = file_parts[0]
    if pred_rand_rob == "predictions":
        pred_setting = "predictions"
    elif pred_rand_rob == "randomization":
        pred_setting = "randomize-" + "-".join(file_parts[1:-2])
    elif pred_rand_rob == "robustness":
        pred_setting = "-".join(file_parts[:2])
    else:
        raise ValueError(f"Unknown prediction setting: {pred_rand_rob}")
    split = "_".join(os.path.basename(pred_file).split(".")[0].split("_")[-2:])
    return f"{model_name}_{pred_setting}_{test_mode}_{split}"


def evaluate_per_group(
    df, group_by, norm_group_eval_results, eval_results_per_group, model
):
    """
    Evaluate the predictions per group.
    :param df:
    :param group_by:
    :param norm_group_eval_results:
    :param eval_results_per_group:
    :param model:
    :return:
    """
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
    """
    Compute the evaluation metrics per group.
    :param df:
    :param return_df:
    :param group_by:
    :param model:
    :return:
    """
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


def write_results(
    path_out, eval_results, eval_results_per_drug, eval_results_per_cl, t_vs_p
):
    """
    Write the results to csv files.
    :param path_out:
    :param eval_results:
    :param eval_results_per_drug:
    :param eval_results_per_cl:
    :param t_vs_p:
    :return:
    """
    eval_results.to_csv(f"{path_out}evaluation_results.csv", index=True)
    if eval_results_per_drug is not None:
        eval_results_per_drug.to_csv(
            f"{path_out}evaluation_results_per_drug.csv", index=True
        )
    if eval_results_per_cl is not None:
        eval_results_per_cl.to_csv(
            f"{path_out}evaluation_results_per_cl.csv", index=True
        )
    t_vs_p.to_csv(f"{path_out}true_vs_pred.csv", index=True)


def create_index_html(custom_id: str, test_modes: List[str], prefix_results: str):
    """
    Create the index.html file.
    :param custom_id:
    :param test_modes:
    :param prefix_results:
    :return:
    """
    # copy images to the results directory
    file_to_copy = [
        "favicon.png",
        "nf-core-drugresponseeval_logo_light.png",
    ]
    for file in file_to_copy:
        file_path = os.path.join(
            str(importlib_resources.files("drevalpy")),
            "visualization",
            "style_utils",
            file,
        )
        shutil.copyfile(file_path, os.path.join(prefix_results, file))

    layout_path = os.path.join(
        str(importlib_resources.files("drevalpy")),
        "visualization",
        "style_utils",
        "index_layout.html",
    )
    idx_html_path = os.path.join(prefix_results, "index.html")
    with open(idx_html_path, "w", encoding="utf-8") as f:
        parse_layout(f=f, path_to_layout=layout_path)
        f.write('<div class="main">\n')
        f.write(
            '<img src="nf-core-drugresponseeval_logo_light.png" '
            'width="364px" height="100px" alt="Logo">\n'
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
                str(importlib_resources.files("drevalpy")),
                "visualization",
                "style_utils",
                f"{lpo_lco_ldo}.png",
            )
            shutil.copyfile(
                img_path, os.path.join(prefix_results, f"{lpo_lco_ldo}.png")
            )
            f.write(
                f'<a href="{lpo_lco_ldo}.html" target="_blank"><img src="{lpo_lco_ldo}.png" '
                f'style="width:300px;height:300px;"></a>\n'
            )
        f.write("</div>\n")
        f.write("</div>\n")
        f.write("</body>\n")
        f.write("</html>\n")


def create_html(run_id: str, lpo_lco_ldo: str, files: list, prefix_results: str):
    """
    Create the html file for the given test mode.
    :param run_id:
    :param lpo_lco_ldo:
    :param files:
    :param prefix_results:
    :return:
    """
    page_layout = os.path.join(
        str(importlib_resources.files("drevalpy")),
        "visualization/style_utils/page_layout.html",
    )
    html_path = os.path.join(prefix_results, f"{lpo_lco_ldo}.html")

    with open(html_path, "w", encoding="utf-8") as f:
        parse_layout(f=f, path_to_layout=page_layout)
        f.write(f"<h1>Results for {run_id}: {lpo_lco_ldo}</h1>\n")

        # Critical difference plot
        f = CriticalDifferencePlot.write_to_html(lpo_lco_ldo=lpo_lco_ldo, f=f)

        # Violin plots
        f = VioHeat.write_to_html(
            lpo_lco_ldo=lpo_lco_ldo, f=f, files=files, plot="Violin"
        )

        # Heatmaps
        f = VioHeat.write_to_html(
            lpo_lco_ldo=lpo_lco_ldo, f=f, files=files, plot="Heatmap"
        )

        # Regression plots
        f = RegressionSliderPlot.write_to_html(
            lpo_lco_ldo=lpo_lco_ldo, f=f, files=files
        )

        # Correlation comparison: Drug
        f = CorrelationComparisonScatter.write_to_html(
            lpo_lco_ldo=lpo_lco_ldo, f=f, files=files
        )

        # Evaluation results tables
        f = HTMLTable.write_to_html(
            lpo_lco_ldo=lpo_lco_ldo, f=f, files=files, prefix=prefix_results
        )

        f.write("</div>\n")
        f.write("</body>\n")
        f.write("</html>\n")
