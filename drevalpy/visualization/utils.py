"""Utility functions for the visualization part of the package."""

import os
import pathlib
import shutil
from typing import TextIO

import importlib_resources
import pandas as pd

from ..datasets.dataset import DrugResponseDataset
from ..datasets.splits import MANIFEST_FILENAME, read_split_manifest
from ..evaluation import AVAILABLE_METRICS, evaluate
from ..utils._pipeline_function import pipeline_function
from . import (
    ComparisonScatter,
    CriticalDifferencePlot,
    CrossStudyTables,
    Heatmap,
    RegressionSliderPlot,
    VioHeat,
    Violin,
)
from .prep_results_format import (
    add_index_columns_from_model,
    apply_mean_effects_normalization,
    enrich_eval_results_per_cell_line,
    enrich_eval_results_per_drug,
    enrich_true_vs_pred,
    load_drug_and_cell_line_metadata,
)
from .result_csv_discovery import discover_result_csv_files


def _discover_result_csv_files(result_dir: pathlib.Path, dataset: str) -> list[pathlib.Path]:
    """Backward-compatible wrapper around :func:`discover_result_csv_files`."""
    return discover_result_csv_files(result_dir, dataset)


def create_output_directories(result_path: pathlib.Path, custom_id: str) -> None:
    """Create visualization output subdirectories if missing.

    Args:
        result_path: Root results directory.
        custom_id: Run identifier subdirectory name.
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


def _resolve_result_test_mode(result_dir: pathlib.Path, dataset: str, split_label: str) -> str:
    """
    Resolve the semantic test mode for a result directory.

    Custom split labels such as ``scaling-lco`` are path labels only. When a split
    manifest is present, use its ``test_mode`` field; otherwise fall back to the label.

    :param result_dir: root results directory
    :param dataset: dataset name, e.g., GDSC2
    :param split_label: directory label under the dataset folder
    :returns: semantic test mode used for evaluation and plotting
    """
    manifest_path = result_dir / dataset / split_label / "splits" / MANIFEST_FILENAME
    manifest = read_split_manifest(manifest_path)
    if manifest is not None:
        test_mode = manifest.get("test_mode")
        if isinstance(test_mode, str) and test_mode.strip():
            return test_mode.strip()
    return split_label


def parse_results(path_to_results: str, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse experiment outputs and compute evaluation metrics.

    Args:
        path_to_results: Directory containing experiment result CSVs.
        dataset: Dataset name subdirectory (for example ``"GDSC2"``).

    Returns:
        Tuple of overall, per-drug, per-cell-line evaluation tables, and true
        versus predicted values.
    """
    print("Generating result tables ...")
    result_dir = pathlib.Path(path_to_results)
    result_files = _discover_result_csv_files(result_dir, dataset)

    # inititalize dictionaries to store the evaluation results
    evaluation_results = None
    evaluation_results_per_drug = None
    evaluation_results_per_cell_line = None
    true_vs_pred = None

    # read every result file and compute the evaluation metrics
    for file in result_files:
        rel_file = str(os.path.normpath(file.relative_to(result_dir))).replace("\\", "/")
        print(f'Evaluating file: "{rel_file}" ...')
        split_label = file.parent.parent.parent.name
        algorithm = file.parent.parent.name
        test_mode = _resolve_result_test_mode(result_dir, dataset, split_label)
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
    """Evaluate predictions from a single result CSV file.

    Args:
        pred_file: Path to a prediction CSV file.
        test_mode: Evaluation test mode (for example ``"LPO"``).
        model_name: Model or algorithm name.
        dataset_name: Dataset label stored on the loaded ``DrugResponseDataset``.

    Returns:
        Tuple of overall evaluation, per-drug, per-cell-line tables, true versus
        predicted values, and the generated model run name.
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
    """Enrich raw evaluation tables with metadata and normalized metrics.

    Args:
        eval_results: Overall evaluation results.
        eval_results_per_drug: Per-drug evaluation results.
        eval_results_per_cell_line: Per-cell-line evaluation results.
        t_vs_p: True versus predicted values.
        path_data: Dataset root for drug and cell-line metadata files.

    Returns:
        The same four tables after reformatting and normalization.

    Raises:
        ValueError: If ``NaiveMeanEffectsPredictor`` is missing from results.
    """
    print("Getting information about drugs and cell lines ...")
    drug_metadata, cell_line_metadata = load_drug_and_cell_line_metadata(path_data)

    print("Reformatting the evaluation results ...")
    eval_results = add_index_columns_from_model(eval_results)
    eval_results_per_drug = enrich_eval_results_per_drug(eval_results_per_drug, drug_metadata)
    eval_results_per_cell_line = enrich_eval_results_per_cell_line(eval_results_per_cell_line, cell_line_metadata)

    print("Reformatting the true vs. predicted values ...")
    t_vs_p = enrich_true_vs_pred(t_vs_p, drug_metadata, cell_line_metadata)
    eval_results = apply_mean_effects_normalization(eval_results, t_vs_p)

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
    """Compute evaluation metrics per drug or cell-line group.

    Args:
        df: True versus predicted values.
        return_df: Existing results table to append to, or ``None``.
        group_by: Grouping column (``"drug"`` or ``"cell_line"``).
        model: Model run name stored on output rows.

    Returns:
        Evaluation metrics aggregated per group.
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
    """Write evaluation tables to CSV files.

    Args:
        path_out: Output directory (for example ``results/my_run/``).
        eval_results: Overall evaluation results.
        eval_results_per_drug: Per-drug evaluation results.
        eval_results_per_cl: Per-cell-line evaluation results.
        t_vs_p: True versus predicted values.
    """
    eval_results.to_csv(f"{path_out}evaluation_results.csv", index=True)
    if eval_results_per_drug is not None:
        eval_results_per_drug.to_csv(f"{path_out}evaluation_results_per_drug.csv", index=True)
    if eval_results_per_cl is not None:
        eval_results_per_cl.to_csv(f"{path_out}evaluation_results_per_cl.csv", index=True)
    t_vs_p.to_csv(f"{path_out}true_vs_pred.csv", index=True)


@pipeline_function
def create_index_html(custom_id: str, test_modes: list[str], prefix_results: str) -> None:
    """Create the report index HTML page.

    Args:
        custom_id: Run identifier (for example ``my_run``).
        test_modes: Test modes to link from the index page.
        prefix_results: Directory containing per-mode HTML reports.
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
    """Create the per-test-mode HTML report page.

    Args:
        run_id: Run identifier shown in the page title.
        test_mode: Test mode for this report (for example ``"LPO"``).
        files: List of generated artifact filenames in the run directory.
        prefix_results: Directory containing report assets and subfolders.
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
    """Draw all per-algorithm plots for one test mode.

    Args:
        model: Model or algorithm name.
        ev_res: Overall evaluation results.
        ev_res_per_drug: Per-drug evaluation results.
        ev_res_per_cell_line: Per-cell-line evaluation results.
        t_vs_p: True versus predicted values.
        test_mode: Evaluation test mode.
        custom_id: Run identifier for output paths.
        result_path: Root results directory.
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
