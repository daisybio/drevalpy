"""Utility functions for the visualization part of the package."""

import os
import pathlib
import re
import shutil
from typing import Optional, TextIO

import importlib_resources
import pandas as pd

from ..datasets.dataset import DrugResponseDataset
from ..evaluation import AVAILABLE_METRICS, evaluate
from ..pipeline_function import pipeline_function
from .corr_comp_scatter import CorrelationComparisonScatter
from .critical_difference_plot import CriticalDifferencePlot
from .html_tables import HTMLTable
from .regression_slider_plot import RegressionSliderPlot
from .vioheat import VioHeat


def _parse_layout(f: TextIO, path_to_layout: str) -> None:
    """
    Parse the layout file and write it to the output file.

    :param f: file to write to
    :param path_to_layout: path to the layout file
    """
    with open(path_to_layout, encoding="utf-8") as layout_f:
        layout = layout_f.readlines()
    if path_to_layout.endswith("index_layout.html"):
        # remove the last 2 lines (</body>, </html>)
        layout = layout[:-2]
    else:
        # remove the last 3 lines (</div>, </body>, </html>)
        layout = layout[:-3]
    f.write("".join(layout))


def parse_results(path_to_results: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the results from the given directory.

    :param path_to_results: path to the results directory
    :returns: evaluation results, evaluation results per drug, evaluation results per cell line, and true vs. predicted
        values
    """
    print("Generating result tables ...")
    # generate list of all result files
    result_dir = pathlib.Path(path_to_results)
    result_files = list(result_dir.rglob("*.csv"))
    # filter for all files that follow this pattern:
    # result_dir/*/{predictions|cross_study|randomization|robustness}/*.csv
    # Convert the path to a forward-slash version for the regex (for Windows)
    result_dir_str = str(result_dir).replace("\\", "/")
    pattern = re.compile(
        rf"{result_dir_str}/(LPO|LCO|LDO)/[^/]+/(predictions|cross_study|randomization|robustness)/.*\.csv$"
    )
    result_files = [file for file in result_files if pattern.match(str(file).replace("\\", "/"))]

    # inititalize dictionaries to store the evaluation results
    evaluation_results = None
    evaluation_results_per_drug = None
    evaluation_results_per_cell_line = None
    true_vs_pred = None

    # read every result file and compute the evaluation metrics
    for file in result_files:
        rel_file = str(os.path.normpath(file.relative_to(result_dir))).replace("\\", "/")
        print(f'Evaluating file: "{rel_file}" ...')
        file_parts = rel_file.split("/")
        lpo_lco_ldo = file_parts[0]
        algorithm = file_parts[1]
        (
            overall_eval,
            eval_results_per_drug,
            eval_results_per_cl,
            t_vs_p,
            model_name,
        ) = evaluate_file(pred_file=file, test_mode=lpo_lco_ldo, model_name=algorithm)

        evaluation_results = (
            overall_eval if evaluation_results is None else pd.concat([evaluation_results, overall_eval])
        )
        true_vs_pred = t_vs_p if true_vs_pred is None else pd.concat([true_vs_pred, t_vs_p])

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


@pipeline_function
def evaluate_file(
    pred_file: pathlib.Path, test_mode: str, model_name: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Evaluate the predictions from the final models.

    :param pred_file: path to the prediction file
    :param test_mode: test mode, e.g., LPO
    :param model_name: model name, e.g., SimpleNeuralNetwork
    :return: evaluation results, evaluation results per drug, evaluation results per cell line, true vs. predicted
        values, and model name
    """
    print("Parsing file:", os.path.normpath(pred_file))
    dataset = DrugResponseDataset.from_csv(pred_file)

    model = _generate_model_names(test_mode=test_mode, model_name=model_name, pred_file=pred_file)

    # overall evaluation
    overall_eval = {model: evaluate(dataset, list(AVAILABLE_METRICS.keys()))}

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
    norm_drug_eval_results: dict[str, dict[str, float]] = {}
    norm_cl_eval_results: dict[str, dict[str, float]] = {}

    if "LPO" in model or "LCO" in model:
        norm_drug_eval_results, evaluation_results_per_drug = _evaluate_per_group(
            df=true_vs_pred,
            group_by="drug",
            norm_group_eval_results=norm_drug_eval_results,
            eval_results_per_group=evaluation_results_per_drug,
            model=model,
        )
    if "LPO" in model or "LDO" in model:
        norm_cl_eval_results, evaluation_results_per_cl = _evaluate_per_group(
            df=true_vs_pred,
            group_by="cell_line",
            norm_group_eval_results=norm_cl_eval_results,
            eval_results_per_group=evaluation_results_per_cl,
            model=model,
        )
    overall_eval = pd.DataFrame.from_dict(overall_eval, orient="index")
    if len(norm_drug_eval_results) > 0:
        overall_eval = _concat_results(norm_drug_eval_results, "drug", overall_eval)
    if len(norm_cl_eval_results) > 0:
        overall_eval = _concat_results(norm_cl_eval_results, "cell_line", overall_eval)

    return (
        overall_eval,
        evaluation_results_per_drug,
        evaluation_results_per_cl,
        true_vs_pred,
        model,
    )


def _concat_results(norm_group_res: dict[str, dict[str, float]], group_by: str, eval_res: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate the normalized group results to the evaluation results.

    :param norm_group_res: dictionary with the normalized group results, key: model name, value: evaluation results
    :param group_by: either cell line or drug
    :param eval_res: overall dataframe
    :returns: overall dataframe extended by the normalized group results
    """
    norm_group_df = pd.DataFrame.from_dict(norm_group_res, orient="index")
    # append 'group normalized ' to the column names
    norm_group_df.columns = [f"{col}: {group_by} normalized" for col in norm_group_df.columns]
    eval_res = pd.concat([eval_res, norm_group_df], axis=1)
    return eval_res


@pipeline_function
def prep_results(
    eval_results: pd.DataFrame,
    eval_results_per_drug: pd.DataFrame,
    eval_results_per_cell_line: pd.DataFrame,
    t_vs_p: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare the results by introducing new columns for algorithm, randomization, setting, split, CV_split.

    :param eval_results: evaluation results
    :param eval_results_per_drug: evaluation results per drug
    :param eval_results_per_cell_line: evaluation results per cell line
    :param t_vs_p: true vs. predicted values
    :returns: the same dataframes with new columns
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
        eval_results_per_drug[["algorithm", "rand_setting", "LPO_LCO_LDO", "split", "CV_split"]] = (
            eval_results_per_drug["model"].str.split("_", expand=True)
        )
    if eval_results_per_cell_line is not None:
        eval_results_per_cell_line[["algorithm", "rand_setting", "LPO_LCO_LDO", "split", "CV_split"]] = (
            eval_results_per_cell_line["model"].str.split("_", expand=True)
        )
    t_vs_p[["algorithm", "rand_setting", "LPO_LCO_LDO", "split", "CV_split"]] = t_vs_p["model"].str.split(
        "_", expand=True
    )

    return (
        eval_results,
        eval_results_per_drug,
        eval_results_per_cell_line,
        t_vs_p,
    )


def _generate_model_names(test_mode: str, model_name: str, pred_file: pathlib.Path) -> str:
    """
    Generate the model names based on the prediction file.

    :param test_mode: test mode, e.g., LPO
    :param model_name: model name, e.g., SimpleNeuralNetwork
    :param pred_file: file containing the predictions
    :returns: unique name of run = {model_name}_{pred_setting}_{test_mode}_{split}
    :raises ValueError: if the prediction setting is unknown
    """
    file_parts = os.path.basename(pred_file).split("_")
    pred_rand_rob = file_parts[0]
    if pred_rand_rob == "predictions":
        pred_setting = "predictions"
    elif pred_rand_rob == "randomization":
        pred_setting = "randomize-" + "-".join(file_parts[1:-2])
    elif pred_rand_rob == "robustness":
        pred_setting = "-".join(file_parts[:2])
    elif pred_rand_rob == "cross":
        pred_setting = "cross-study-" + file_parts[2]
    else:
        raise ValueError(f"Unknown prediction setting: {pred_rand_rob}")
    split = "_".join(os.path.basename(pred_file).split(".")[0].split("_")[-2:])
    return f"{model_name}_{pred_setting}_{test_mode}_{split}"


def _evaluate_per_group(
    df: pd.DataFrame,
    group_by: str,
    norm_group_eval_results: dict[str, dict[str, float]],
    eval_results_per_group: Optional[pd.DataFrame],
    model: str,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """
    Evaluate the predictions per group.

    :param df: true vs. predicted values
    :param group_by: cell line or drug
    :param norm_group_eval_results: dictionary to store the normalized group evaluation results
    :param eval_results_per_group: evaluation results per group
    :param model: model name
    :returns: dictionary with the normalized group evaluation results and the evaluation results per group
    """
    # calculate the mean of y_true per drug
    print(f"Calculating {group_by}-wise evaluation measures …")
    df[f"mean_y_true_per_{group_by}"] = df.groupby(group_by)["y_true"].transform("mean")
    norm_df = df.copy()
    norm_df["y_true"] = norm_df["y_true"] - norm_df[f"mean_y_true_per_{group_by}"]
    norm_df["y_pred"] = norm_df["y_pred"] - norm_df[f"mean_y_true_per_{group_by}"]
    norm_group_eval_results[model] = evaluate(
        DrugResponseDataset(
            response=norm_df["y_true"].to_numpy(),
            cell_line_ids=norm_df["cell_line"].to_numpy(),
            drug_ids=norm_df["drug"].to_numpy(),
            predictions=norm_df["y_pred"].to_numpy(),
        ),
        list(AVAILABLE_METRICS.keys() - {"MSE", "RMSE", "MAE"}),
    )
    # evaluation per group
    eval_results_per_group = compute_evaluation(df, eval_results_per_group, group_by, model)
    return norm_group_eval_results, eval_results_per_group


def compute_evaluation(df: pd.DataFrame, return_df: pd.DataFrame | None, group_by: str, model: str) -> pd.DataFrame:
    """
    Compute the evaluation metrics per group.

    :param df: true vs. predicted values with mean_y_true_per_{group_by} column
    :param return_df: DataFrame to store the results
    :param group_by: either cell line or drug
    :param model: model name
    :returns: dataframe with the evaluation results per group
    """
    result_per_group = df.groupby(group_by)[["y_true", "cell_line", "drug", "y_pred"]].apply(
        lambda x: evaluate(
            DrugResponseDataset(
                response=x["y_true"].to_numpy(),
                cell_line_ids=x["cell_line"].to_numpy(),
                drug_ids=x["drug"].to_numpy(),
                predictions=x["y_pred"].to_numpy(),
            ),
            list(AVAILABLE_METRICS.keys()),
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


@pipeline_function
def write_results(
    path_out: str,
    eval_results: pd.DataFrame,
    eval_results_per_drug: pd.DataFrame,
    eval_results_per_cl: pd.DataFrame,
    t_vs_p: pd.DataFrame,
) -> None:
    """
    Write the results to csv files.

    :param path_out: path to the output directory, e.g., results/my_run/
    :param eval_results: evaluation results
    :param eval_results_per_drug: evaluation results per drug
    :param eval_results_per_cl: evaluation results per cell line
    :param t_vs_p: true vs. predicted values
    """
    eval_results.to_csv(f"{path_out}evaluation_results.csv", index=True)
    if eval_results_per_drug is not None:
        eval_results_per_drug.to_csv(f"{path_out}evaluation_results_per_drug.csv", index=True)
    if eval_results_per_cl is not None:
        eval_results_per_cl.to_csv(f"{path_out}evaluation_results_per_cl.csv", index=True)
    t_vs_p.to_csv(f"{path_out}true_vs_pred.csv", index=True)


@pipeline_function
def create_index_html(custom_id: str, test_modes: list[str], prefix_results: str) -> None:
    """
    Create the index.html file.

    :param custom_id: custom id for the results, e.g., my_run
    :param test_modes: list of test modes, e.g., ["LPO", "LCO", "LDO"]
    :param prefix_results: path to the results directory, e.g., results/my_run
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
        _parse_layout(f=f, path_to_layout=layout_path)
        f.write('<div class="main">\n')
        f.write('<img src="nf-core-drugresponseeval_logo_light.png" ' 'width="364px" height="100px" alt="Logo">\n')
        f.write(f"<h1>Results for {custom_id}</h1>\n")
        f.write("<h2>Available settings</h2>\n")
        f.write('<div style="display: inline-block;">\n')
        f.write("<p>Click on the images to open the respective report in a new tab.</p>\n")

        test_modes.sort()
        for lpo_lco_ldo in test_modes:
            img_path = os.path.join(
                str(importlib_resources.files("drevalpy")),
                "visualization",
                "style_utils",
                f"{lpo_lco_ldo}.png",
            )
            shutil.copyfile(img_path, os.path.join(prefix_results, f"{lpo_lco_ldo}.png"))
            f.write(
                f'<a href="{lpo_lco_ldo}.html" target="_blank"><img src="{lpo_lco_ldo}.png" '
                f'style="width:300px;height:300px;"></a>\n'
            )
        f.write("</div>\n")
        f.write("</div>\n")
        f.write("</body>\n")
        f.write("</html>\n")


@pipeline_function
def create_html(run_id: str, lpo_lco_ldo: str, files: list, prefix_results: str) -> None:
    """
    Create the html file for the given test mode, e.g., LPO.html.

    :param run_id: custom id for the results, e.g., my_run
    :param lpo_lco_ldo: test mode, e.g., LPO
    :param files: list of files in the results directory
    :param prefix_results: path to the results directory, e.g., results/my_run
    """
    page_layout = os.path.join(
        str(importlib_resources.files("drevalpy")),
        "visualization/style_utils/page_layout.html",
    )
    html_path = os.path.join(prefix_results, f"{lpo_lco_ldo}.html")

    with open(html_path, "w", encoding="utf-8") as f:
        _parse_layout(f=f, path_to_layout=page_layout)
        f.write(f"<h1>Results for {run_id}: {lpo_lco_ldo}</h1>\n")

        # Critical difference plot
        f = CriticalDifferencePlot.write_to_html(lpo_lco_ldo=lpo_lco_ldo, f=f)

        # Violin plots
        f = VioHeat.write_to_html(lpo_lco_ldo=lpo_lco_ldo, f=f, files=files, plot="Violin")

        # Heatmaps
        f = VioHeat.write_to_html(lpo_lco_ldo=lpo_lco_ldo, f=f, files=files, plot="Heatmap")

        # Regression plots
        f = RegressionSliderPlot.write_to_html(lpo_lco_ldo=lpo_lco_ldo, f=f, files=files)

        # Correlation comparison: Drug
        f = CorrelationComparisonScatter.write_to_html(lpo_lco_ldo=lpo_lco_ldo, f=f, files=files)

        # Evaluation results tables
        f = HTMLTable.write_to_html(lpo_lco_ldo=lpo_lco_ldo, f=f, files=files, prefix=prefix_results)

        f.write("</div>\n")
        f.write("</body>\n")
        f.write("</html>\n")
