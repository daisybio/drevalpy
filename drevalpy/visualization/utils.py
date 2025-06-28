"""Utility functions for the visualization part of the package."""

import os
import pathlib
import re
import shutil
from typing import TextIO

import importlib_resources
import numpy as np
import pandas as pd

from ..datasets.dataset import DrugResponseDataset
from ..evaluation import AVAILABLE_METRICS, evaluate
from ..models.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER
from ..pipeline_function import pipeline_function
from . import (
    ComparisonScatter,
    CriticalDifferencePlot,
    CrossStudyTables,
    Heatmap,
    RegressionSliderPlot,
    VioHeat,
    Violin,
)


def create_output_directories(result_path: pathlib.Path, custom_id: str) -> None:
    """
    If they do not exist yet, make directories for the visualization files.

    :param result_path: path to the results
    :param custom_id: run id passed via command line
    """
    for dir in [
        "violin_plots",
        "heatmaps",
        "regression_plots",
        "comp_scatter",
        "html_tables",
        "critical_difference_plots",
    ]:
        os.makedirs(pathlib.Path(result_path / custom_id / dir), exist_ok=True)


def _parse_layout(f: TextIO, path_to_layout: str, test_mode: str) -> None:
    """
    Parse the layout file and write it to the output file.

    :param f: file to write to
    :param path_to_layout: path to the layout file
    :param test_mode: test mode, e.g., LPO
    """
    with open(path_to_layout, encoding="utf-8") as layout_f:
        layout = layout_f.readlines()
    if path_to_layout.endswith("index_layout.html"):
        # remove the last 2 lines (</body>, </html>)
        layout = layout[:-2]
    else:
        # remove the last 3 lines (</div>, </body>, </html>)
        layout = layout[:-3]
        # replace LPOLCOLDO with the test mode
        layout = [line.replace("LPOLCOLDO", test_mode) for line in layout]
    f.write("".join(layout))


def parse_results(path_to_results: str, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the results from the given directory.

    :param path_to_results: path to the results directory
    :param dataset: dataset name, e.g., GDSC2
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
        rf"{result_dir_str}/{dataset}/"
        r"(LPO|LCO|LDO|LTO)/[^/]+/(predictions|cross_study|randomization|robustness)/.*\.csv$"
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
        dataset = file_parts[0]
        test_mode = file_parts[1]
        algorithm = file_parts[2]
        (
            overall_eval,
            eval_results_per_drug,
            eval_results_per_cl,
            t_vs_p,
            model_name,
        ) = evaluate_file(pred_file=file, test_mode=test_mode, model_name=algorithm)

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
    pred_file: pathlib.Path, test_mode: str, model_name: str, dataset_name: str = "NO_DATASET_NAME"
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame, str]:
    """
    Evaluate the predictions from the final models.

    :param pred_file: path to the prediction file
    :param test_mode: test mode, e.g., LPO
    :param model_name: model name, e.g., SimpleNeuralNetwork
    :param dataset_name: name of the dataset, e.g., GDSC2
    :return: evaluation results, evaluation results per drug, evaluation results per cell line, true vs. predicted
        values, and model name
    """
    print("Parsing file:", os.path.normpath(pred_file))
    dataset = DrugResponseDataset.from_csv(input_file=pred_file, dataset_name=dataset_name)

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

    if "LPO" in model or "LDO" in model:
        evaluation_results_per_drug = _evaluate_per_group(
            df=true_vs_pred,
            group_by="drug",
            eval_results_per_group=evaluation_results_per_drug,
            model=model,
        )
    if "LPO" in model or "LCO" in model or "LTO" in model:
        evaluation_results_per_cl = _evaluate_per_group(
            df=true_vs_pred,
            group_by="cell_line",
            eval_results_per_group=evaluation_results_per_cl,
            model=model,
        )
    overall_eval = pd.DataFrame.from_dict(overall_eval, orient="index")

    return (
        overall_eval,
        evaluation_results_per_drug,
        evaluation_results_per_cl,
        true_vs_pred,
        model,
    )


@pipeline_function
def prep_results(
    eval_results: pd.DataFrame,
    eval_results_per_drug: pd.DataFrame,
    eval_results_per_cell_line: pd.DataFrame,
    t_vs_p: pd.DataFrame,
    path_data: pathlib.Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare the results by introducing new columns for algorithm, randomization, test_mode, split, CV_split.

    :param eval_results: evaluation results
    :param eval_results_per_drug: evaluation results per drug
    :param eval_results_per_cell_line: evaluation results per cell line
    :param t_vs_p: true vs. predicted values
    :param path_data: path to the data
    :returns: the same dataframes with new columns
    :raises ValueError: if NaiveMeanEffectsPredictor is not found in the evaluation results
    """
    # get metadata
    print("Getting information about drugs and cell lines ...")
    drug_metadata: dict[str, str] = dict()
    cell_line_metadata: dict[str, str] = dict()
    for root, _, files in os.walk(path_data):
        for file in files:
            if file == "drug_names.csv":
                drug_names = pd.read_csv(os.path.join(root, file), index_col=0)
                # make index to str
                drug_names.index = drug_names.index.astype(str)
                # index: pubchem_id, column: drug_name
                drug_metadata.update(zip(drug_names.index, drug_names["drug_name"]))
            elif file == "cell_line_names.csv":
                cell_line_names = pd.read_csv(os.path.join(root, file), index_col=0)
                # index: cellosaurus_id, column: cell_line_name
                cell_line_metadata.update(zip(cell_line_names[CELL_LINE_IDENTIFIER], cell_line_names.index))

    # add variables
    # split the index by "_" into: algorithm, randomization, test_mode, split, CV_split
    print("Reformatting the evaluation results ...")
    new_columns = eval_results.index.str.split("_", expand=True).to_frame()
    new_columns.columns = [
        "algorithm",
        "rand_setting",
        "test_mode",
        "split",
        "CV_split",
    ]
    new_columns.index = eval_results.index
    eval_results = pd.concat([new_columns.drop("split", axis=1), eval_results], axis=1)
    if eval_results_per_drug is not None:
        print("Reformatting the evaluation results per drug ...")
        eval_results_per_drug[["algorithm", "rand_setting", "test_mode", "split", "CV_split"]] = eval_results_per_drug[
            "model"
        ].str.split("_", expand=True)
        all_drugs = [drug_metadata[drug] for drug in eval_results_per_drug["drug"]]
        eval_results_per_drug["drug_name"] = all_drugs
        # rename drug to pubchem_id
        eval_results_per_drug = eval_results_per_drug.rename(columns={"drug": DRUG_IDENTIFIER})
    if eval_results_per_cell_line is not None:
        print("Reformatting the evaluation results per cell line ...")
        eval_results_per_cell_line[["algorithm", "rand_setting", "test_mode", "split", "CV_split"]] = (
            eval_results_per_cell_line["model"].str.split("_", expand=True)
        )
        all_cello_ids = [cell_line_metadata[cell_line] for cell_line in eval_results_per_cell_line["cell_line"]]
        eval_results_per_cell_line["cellosaurus_id"] = all_cello_ids
        eval_results_per_cell_line = eval_results_per_cell_line.rename(columns={"cell_line": CELL_LINE_IDENTIFIER})

    print("Reformatting the true vs. predicted values ...")
    t_vs_p[["algorithm", "rand_setting", "test_mode", "split", "CV_split"]] = t_vs_p["model"].str.split(
        "_", expand=True
    )
    t_vs_p = t_vs_p.drop("split", axis=1)
    all_drugs = [drug_metadata[drug] for drug in t_vs_p["drug"]]
    t_vs_p["drug_name"] = all_drugs
    all_cello_ids = [cell_line_metadata[cell_line] for cell_line in t_vs_p["cell_line"]]
    t_vs_p["cellosaurus_id"] = all_cello_ids
    t_vs_p = t_vs_p.rename(columns={"cell_line": CELL_LINE_IDENTIFIER, "drug": DRUG_IDENTIFIER})
    t_vs_p[DRUG_IDENTIFIER] = t_vs_p[DRUG_IDENTIFIER].astype(str)

    if "NaiveMeanEffectsPredictor" in eval_results["algorithm"].unique():
        eval_results = _normalize_metrics_by_mean_effects(
            evaluation_results=eval_results,
            true_vs_pred=t_vs_p,
        )
    else:
        raise ValueError(
            "NaiveMeanEffectsPredictor not found in evaluation results. "
            "Please check if the evaluation was run correctly."
        )

    return (
        eval_results,
        eval_results_per_drug,
        eval_results_per_cell_line,
        t_vs_p,
    )


def _normalize_metrics_by_mean_effects(
    evaluation_results: pd.DataFrame,
    true_vs_pred: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize the y_true and y_pred values by the predictions of the NaiveMeanEffectsPredictor.

    Then recalculate the metrics.
    :param evaluation_results: results of the evaluation
    :param true_vs_pred: all true vs. predicted values
    :return: modified evaluation results
    """
    eval_results_mod = {}
    naive_mean_effects_dict = {}
    for rand_setting in evaluation_results["rand_setting"].unique():
        for test_mode in evaluation_results["test_mode"].unique():
            naive_mean_effects_dict[f"{test_mode}_{rand_setting}"] = true_vs_pred[
                (true_vs_pred["algorithm"] == "NaiveMeanEffectsPredictor")
                & (true_vs_pred["rand_setting"] == rand_setting)
                & (true_vs_pred["test_mode"] == test_mode)
            ]
    # do this: per algorithm, per rand test_mode, per test_mode, per CV split
    for algorithm in evaluation_results["algorithm"].unique():
        for rand_setting in evaluation_results["rand_setting"].unique():
            for test_mode in evaluation_results["test_mode"].unique():

                setting_subset = true_vs_pred[
                    (true_vs_pred["algorithm"] == algorithm)
                    & (true_vs_pred["rand_setting"] == rand_setting)
                    & (true_vs_pred["test_mode"] == test_mode)
                ]
                if setting_subset.empty:
                    continue
                naive_mean_effects = naive_mean_effects_dict[f"{test_mode}_{rand_setting}"]
                naive_mean_effects = naive_mean_effects[["drug_name", "cell_line_name", "CV_split", "y_pred"]]
                naive_mean_effects = naive_mean_effects.rename(columns={"y_pred": "y_pred_naive"})
                setting_subset = setting_subset[["drug_name", "cell_line_name", "CV_split", "y_true", "y_pred"]]
                setting_subset = setting_subset.merge(
                    naive_mean_effects, on=["drug_name", "cell_line_name", "CV_split"], how="left"
                )
                setting_subset["y_true"] = setting_subset["y_true"] - setting_subset["y_pred_naive"]
                setting_subset["y_pred"] = setting_subset["y_pred"] - setting_subset["y_pred_naive"]
                for cv_split in setting_subset["CV_split"].unique():
                    setting_subset_cv = setting_subset[setting_subset["CV_split"] == cv_split]
                    dt = DrugResponseDataset(
                        response=setting_subset_cv["y_true"].to_numpy(),
                        cell_line_ids=setting_subset_cv["cell_line_name"].to_numpy(),
                        drug_ids=setting_subset_cv["drug_name"].to_numpy(),
                        predictions=setting_subset_cv["y_pred"].to_numpy(),
                    )
                    res = evaluate(
                        dataset=dt,
                        metric=list(AVAILABLE_METRICS.keys() - {"MAE", "MSE", "RMSE"}),
                    )
                    eval_results_mod[f"{algorithm}_{rand_setting}_{test_mode}_split_{cv_split}"] = res
    mod_table = pd.DataFrame.from_dict(eval_results_mod, orient="index")
    mod_table.columns = [f"{col}: normalized" for col in mod_table.columns]
    evaluation_results = evaluation_results.merge(mod_table, left_index=True, right_index=True)
    return evaluation_results


def _generate_model_names(test_mode: str, model_name: str, pred_file: pathlib.Path) -> str:
    """
    Generate the model names based on the prediction file.

    :param test_mode: test mode, e.g., LPO
    :param model_name: model name, e.g., SimpleNeuralNetwork
    :param pred_file: file containing the predictions
    :returns: unique name of run = {model_name}_{pred_setting}_{test_mode}_{split}
    :raises ValueError: if the prediction test_mode is unknown
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
        raise ValueError(f"Unknown prediction test_mode: {pred_rand_rob}")
    split = "_".join(os.path.basename(pred_file).split(".")[0].split("_")[-2:])
    return f"{model_name}_{pred_setting}_{test_mode}_{split}"


def _evaluate_per_group(
    df: pd.DataFrame,
    group_by: str,
    eval_results_per_group: pd.DataFrame | None,
    model: str,
) -> pd.DataFrame:
    """
    Evaluate the predictions per group.

    :param df: true vs. predicted values
    :param group_by: cell line or drug
    :param eval_results_per_group: evaluation results per group
    :param model: model name
    :returns: dictionary with the normalized group evaluation results and the evaluation results per group
    """
    # calculate the mean of y_true per drug
    print(f"Calculating {group_by}-wise evaluation measures …")
    # evaluation per group
    eval_results_per_group = compute_evaluation(df, eval_results_per_group, group_by, model)
    return eval_results_per_group


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
        _parse_layout(f=f, path_to_layout=layout_path, test_mode="")
        f.write('<div class="main">\n')
        f.write('<img src="nf-core-drugresponseeval_logo_light.png" ' 'width="364px" height="100px" alt="Logo">\n')
        f.write(f"<h1>Results for {custom_id}</h1>\n")
        f.write("<h2>Available settings</h2>\n")
        f.write('<div style="display: inline-block;">\n')
        f.write("<p>Click on the images to open the respective report in a new tab.</p>\n")

        test_modes.sort()
        for test_mode in test_modes:
            img_path = os.path.join(
                str(importlib_resources.files("drevalpy")),
                "visualization",
                "style_utils",
                f"{test_mode}.png",
            )
            shutil.copyfile(img_path, os.path.join(prefix_results, f"{test_mode}.png"))
            f.write(
                f'<a href="{test_mode}.html" target="_blank"><img src="{test_mode}.png" '
                f'style="width:300px;height:300px;"></a>\n'
            )
        f.write("</div>\n")
        f.write("</div>\n")
        f.write("</body>\n")
        f.write("</html>\n")


def create_html(run_id: str, test_mode: str, files: list, prefix_results: str) -> None:
    """
    Create the html file for the given test mode, e.g., LPO.html.

    :param run_id: custom id for the results, e.g., my_run
    :param test_mode: test mode, e.g., LPO
    :param files: list of files in the results directory
    :param prefix_results: path to the results directory, e.g., results/my_run
    """
    page_layout = os.path.join(
        str(importlib_resources.files("drevalpy")),
        "visualization/style_utils/page_layout.html",
    )
    html_path = os.path.join(prefix_results, f"{test_mode}.html")

    with open(html_path, "w", encoding="utf-8") as f:
        _parse_layout(f=f, path_to_layout=page_layout, test_mode=test_mode)
        f.write(f"<h1>Results for {run_id}: {test_mode}</h1>\n")

        # Critical difference plot
        f = CriticalDifferencePlot.write_to_html(test_mode=test_mode, f=f)

        # Violin plots
        f = VioHeat.write_to_html(test_mode=test_mode, f=f, files=files, plot="Violin")

        # Heatmaps
        f = VioHeat.write_to_html(test_mode=test_mode, f=f, files=files, plot="Heatmap")

        # Regression plots
        f = RegressionSliderPlot.write_to_html(test_mode=test_mode, f=f, files=files)

        # Correlation comparison: Drug
        f = ComparisonScatter.write_to_html(test_mode=test_mode, f=f, files=files)

        # Cross-study evaluation tables
        f = CrossStudyTables.write_to_html(test_mode=test_mode, f=f, files=files, prefix=prefix_results)

        f.write("</div>\n")
        f.write("</body>\n")
        f.write("</html>\n")


def draw_test_mode_plots(
    test_mode: str,
    ev_res: pd.DataFrame,
    ev_res_per_drug: pd.DataFrame | None,
    ev_res_per_cell_line: pd.DataFrame | None,
    custom_id: str,
    path_data: pathlib.Path,
    result_path: pathlib.Path,
) -> np.ndarray:
    """
    Draw all plots for a specific test_mode (LPO, LCO, LDO, LTO).

    :param test_mode: test_mode
    :param ev_res: overall evaluation results
    :param ev_res_per_drug: evaluation results per drug
    :param ev_res_per_cell_line: evaluation results per cell line
    :param custom_id: run id passed via command line
    :param path_data: path to the data
    :param result_path: path to the results
    :returns: list of unique algorithms
    :raises ValueError: if no evaluation results are found for the given test_mode
    """
    if ev_res.empty:
        raise ValueError(
            f"No evaluation results found for test_mode {test_mode}. "
            "Please check if the evaluation was run correctly."
        )
    ev_res_subset = ev_res[ev_res["test_mode"] == test_mode]

    # only draw figures for 'real' predictions comparing all models
    eval_results_preds = ev_res_subset[ev_res_subset["rand_setting"] == "predictions"]
    if eval_results_preds.empty:
        raise ValueError(
            f"No evaluation results found for test_mode {test_mode} with predictions. "
            "Please check if the evaluation was run correctly."
        )

    cd_plot = CriticalDifferencePlot(eval_results_preds=eval_results_preds, metric="MSE")
    cd_plot.draw_and_save(
        out_prefix=f"{result_path}/{custom_id}/critical_difference_plots/",
        out_suffix=test_mode,
    )
    for plt_type in ["violinplot", "heatmap"]:
        if plt_type == "violinplot":
            out_dir = "violin_plots"
        else:
            out_dir = "heatmaps"
        for normalized in [False, True]:
            if normalized:
                out_suffix = f"algorithms_{test_mode}_normalized"
            else:
                out_suffix = f"algorithms_{test_mode}"
            out_plot: Violin | Heatmap
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
                out_prefix=f"{result_path}/{custom_id}/{out_dir}/",
                out_suffix=out_suffix,
            )

    # per group plots
    if test_mode in ("LPO", "LDO"):
        _draw_per_grouping_setting_plots(
            grouping="drug_name",
            ev_res_per_group=ev_res_per_drug,
            test_mode=test_mode,
            custom_id=custom_id,
            result_path=result_path,
        )
    if test_mode in ("LPO", "LCO", "LTO"):
        _draw_per_grouping_setting_plots(
            grouping="cell_line_name",
            ev_res_per_group=ev_res_per_cell_line,
            test_mode=test_mode,
            custom_id=custom_id,
            result_path=result_path,
        )

    # Cross-study evaluation tables
    cross_study_tables = CrossStudyTables(evaluation_metrics=ev_res_subset, path_data=path_data)
    cross_study_tables.draw_and_save(
        out_prefix=f"{result_path}/{custom_id}/html_tables/",
        out_suffix=test_mode,
    )

    return eval_results_preds["algorithm"].unique()


def _draw_per_grouping_setting_plots(
    grouping: str, ev_res_per_group: pd.DataFrame, test_mode: str, custom_id: str, result_path: pathlib.Path
) -> None:
    """
    Draw plots for a specific grouping (drug or cell line) for a specific test_mode (LPO, LCO, LDO).

    :param grouping: drug or cell_line
    :param ev_res_per_group: evaluation results per drug or per cell line
    :param test_mode: test_mode
    :param custom_id: run id passed over command line
    :param result_path: path to the results
    """
    corr_comp = ComparisonScatter(
        df=ev_res_per_group,
        color_by=grouping,
        test_mode=test_mode,
        algorithm="all",
    )
    if corr_comp.name is not None:
        corr_comp.draw_and_save(
            out_prefix=f"{result_path}/{custom_id}/comp_scatter/",
            out_suffix=corr_comp.name,
        )


def draw_algorithm_plots(
    model: str,
    ev_res: pd.DataFrame,
    ev_res_per_drug: pd.DataFrame | None,
    ev_res_per_cell_line: pd.DataFrame | None,
    t_vs_p: pd.DataFrame,
    test_mode: str,
    custom_id: str,
    result_path: pathlib.Path,
) -> None:
    """
    Draw all plots for a specific algorithm.

    :param model: name of the model/algorithm
    :param ev_res: overall evaluation results
    :param ev_res_per_drug: evaluation results per drug
    :param ev_res_per_cell_line: evaluation results per cell line
    :param t_vs_p: true response values vs. predicted response values
    :param test_mode: test_mode
    :param custom_id: run id passed via command line
    :param result_path: path to the results
    """
    eval_results_algorithm = ev_res[(ev_res["test_mode"] == test_mode) & (ev_res["algorithm"] == model)]
    for plt_type in ["violinplot", "heatmap"]:
        if len(eval_results_algorithm["rand_setting"].unique()) < 2:
            # only draw plots if there are predictions and another test_mode (randomization/robustness)
            continue
        out_plot: Violin | Heatmap
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
            out_prefix=f"{result_path}/{custom_id}/{out_dir}/",
            out_suffix=f"{model}_{test_mode}",
        )
    if test_mode in ("LPO", "LDO"):
        _draw_per_grouping_algorithm_plots(
            grouping="drug_name",
            model=model,
            ev_res_per_group=ev_res_per_drug,
            t_v_p=t_vs_p,
            test_mode=test_mode,
            custom_id=custom_id,
            result_path=result_path,
        )
    if test_mode in ("LPO", "LCO", "LTO"):
        _draw_per_grouping_algorithm_plots(
            grouping="cell_line_name",
            model=model,
            ev_res_per_group=ev_res_per_cell_line,
            t_v_p=t_vs_p,
            test_mode=test_mode,
            custom_id=custom_id,
            result_path=result_path,
        )


def _draw_per_grouping_algorithm_plots(
    grouping: str,
    model: str,
    ev_res_per_group: pd.DataFrame,
    t_v_p: pd.DataFrame,
    test_mode: str,
    custom_id: str,
    result_path: pathlib.Path,
):
    """
    Draw plots for a specific grouping (drug or cell line) for a specific algorithm.

    :param grouping: drug or cell_line
    :param model: name of the model/algorithm
    :param ev_res_per_group: evaluation results per drug or per cell line
    :param t_v_p: true response values vs. predicted response values
    :param test_mode: test_mode
    :param custom_id: run id passed via command line
    :param result_path: path to the results
    """
    if len(ev_res_per_group["rand_setting"].unique()) > 1:
        # only draw plots if there are predictions and another test_mode (randomization/robustness)
        comp_scatter = ComparisonScatter(
            df=ev_res_per_group,
            color_by=grouping,
            test_mode=test_mode,
            algorithm=model,
        )
        if comp_scatter.name is not None:
            comp_scatter.draw_and_save(
                out_prefix=f"{result_path}/{custom_id}/comp_scatter/",
                out_suffix=comp_scatter.name,
            )
    for normalize in [False, True]:
        name_suffix = "_normalized" if normalize else ""
        name = f"{test_mode}_{grouping}{name_suffix}"
        regr_slider = RegressionSliderPlot(
            df=t_v_p,
            test_mode=test_mode,
            model=model,
            group_by=grouping,
            normalize=normalize,
        )
        regr_slider.draw_and_save(
            out_prefix=f"{result_path}/{custom_id}/regression_plots/",
            out_suffix=f"{name}_{model}{name_suffix}",
        )
