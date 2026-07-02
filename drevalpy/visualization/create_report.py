"""Generate evaluation reports after running a drug response experiment."""

import os
import pathlib
from collections.abc import Iterable
from typing import Union

import numpy as np
import pandas as pd

from drevalpy.visualization.utils import (
    create_html,
    create_index_html,
    create_output_directories,
    draw_algorithm_plots,
    draw_test_mode_plots,
    parse_results,
    prep_results,
    write_results,
)


def generate_reports_for_test_mode(
    test_mode: str,
    evaluation_results: pd.DataFrame,
    evaluation_results_per_drug: pd.DataFrame,
    evaluation_results_per_cell_line: pd.DataFrame,
    true_vs_pred: pd.DataFrame,
    run_id: str,
    path_data: Union[str, pathlib.Path],
    result_path: Union[str, pathlib.Path],
) -> None:
    """
    Generate reports (plots and HTML) for a single test mode.

    :param test_mode: The test mode to generate reports for.
    :param evaluation_results: Aggregated evaluation results.
    :param evaluation_results_per_drug: Evaluation results per drug.
    :param evaluation_results_per_cell_line: Evaluation results per cell line.
    :param true_vs_pred: True vs predicted values.
    :param run_id: Unique run identifier.
    :param path_data: Path to the dataset directory.
    :param result_path: Path to the results directory.
    """
    path_data = pathlib.Path(path_data)
    result_path = pathlib.Path(result_path)

    print(f"Generating report for {test_mode} ...")
    unique_algos_ndarray = draw_test_mode_plots(
        test_mode=test_mode,
        ev_res=evaluation_results,
        ev_res_per_drug=evaluation_results_per_drug,
        ev_res_per_cell_line=evaluation_results_per_cell_line,
        custom_id=run_id,
        path_data=path_data,
        result_path=result_path,
    )
    unique_algos: Iterable[str] = (
        list(unique_algos_ndarray) if isinstance(unique_algos_ndarray, (np.ndarray, tuple)) else unique_algos_ndarray
    )

    unique_algos_set = set(unique_algos) - {
        "NaiveMeanEffectsPredictor",
        "NaivePredictor",
        "NaiveCellLineMeansPredictor",
        "NaiveTissueMeansPredictor",
        "NaiveDrugMeanPredictor",
    }
    for algorithm in unique_algos_set:
        draw_algorithm_plots(
            model=algorithm,
            ev_res=evaluation_results,
            ev_res_per_drug=evaluation_results_per_drug,
            ev_res_per_cell_line=evaluation_results_per_cell_line,
            t_vs_p=true_vs_pred,
            test_mode=test_mode,
            custom_id=run_id,
            result_path=result_path,
        )

    all_files = []
    for _, _, files in os.walk(f"{result_path}/{run_id}"):
        for file in files:
            if file.endswith("json") or (
                file.endswith(".html") and file not in ["index.html", "LPO.html", "LCO.html", "LDO.html"]
            ):
                all_files.append(file)

    create_html(
        run_id=run_id,
        test_mode=test_mode,
        files=all_files,
        prefix_results=f"{result_path}/{run_id}",
    )


def generate_reports_for_all_test_modes(
    test_modes: list[str],
    evaluation_results: pd.DataFrame,
    evaluation_results_per_drug: pd.DataFrame,
    evaluation_results_per_cell_line: pd.DataFrame,
    true_vs_pred: pd.DataFrame,
    run_id: str,
    path_data: Union[str, pathlib.Path],
    result_path: Union[str, pathlib.Path],
) -> None:
    """
    Generate reports for all test modes.

    :param test_modes: list of test modes to process.
    :param evaluation_results: Aggregated evaluation results.
    :param evaluation_results_per_drug: Evaluation results per drug.
    :param evaluation_results_per_cell_line: Evaluation results per cell line.
    :param true_vs_pred: True vs predicted values.
    :param run_id: Unique run identifier.
    :param path_data: Path to the dataset directory.
    :param result_path: Path to the results directory.
    """
    for test_mode in test_modes:
        generate_reports_for_test_mode(
            test_mode=test_mode,
            evaluation_results=evaluation_results,
            evaluation_results_per_drug=evaluation_results_per_drug,
            evaluation_results_per_cell_line=evaluation_results_per_cell_line,
            true_vs_pred=true_vs_pred,
            run_id=run_id,
            path_data=path_data,
            result_path=result_path,
        )


def create_report(
    run_id: str,
    dataset: str,
    path_data: Union[str, pathlib.Path] = "data",
    result_path: Union[str, pathlib.Path] = "results",
) -> None:
    """
    Render a full evaluation report pipeline.

    :param run_id: Unique run identifier for locating results.
    :param dataset: Dataset name to filter results.
    :param path_data: Path to the dataset directory. Defaults to "data".
    :param result_path: Path to the results directory. Defaults to "results".

    :raises AssertionError: If the folder with the run_id does not exist under result_path.
    """
    path_data = pathlib.Path(path_data).resolve()
    result_path = pathlib.Path(result_path).resolve()

    if not os.path.exists(f"{result_path}/{run_id}"):
        raise AssertionError(f"Folder {result_path}/{run_id} does not exist. The pipeline has to be run first.")

    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = parse_results(path_to_results=f"{result_path}/{run_id}", dataset=dataset)

    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = prep_results(
        evaluation_results, evaluation_results_per_drug, evaluation_results_per_cell_line, true_vs_pred, path_data
    )

    write_results(
        path_out=f"{result_path}/{run_id}/",
        eval_results=evaluation_results,
        eval_results_per_drug=evaluation_results_per_drug,
        eval_results_per_cl=evaluation_results_per_cell_line,
        t_vs_p=true_vs_pred,
    )

    create_output_directories(result_path, run_id)
    test_modes = list(evaluation_results["test_mode"].unique())

    generate_reports_for_all_test_modes(
        test_modes=test_modes,
        evaluation_results=evaluation_results,
        evaluation_results_per_drug=evaluation_results_per_drug,
        evaluation_results_per_cell_line=evaluation_results_per_cell_line,
        true_vs_pred=true_vs_pred,
        run_id=run_id,
        path_data=path_data,
        result_path=result_path,
    )

    create_index_html(
        custom_id=run_id,
        test_modes=test_modes,
        prefix_results=f"{result_path}/{run_id}",
    )


def run_report(
    *,
    run_id: str,
    dataset: str,
    path_data: str = "data",
    result_path: str = "results",
) -> None:
    """Generate HTML report from a standalone experiment run."""
    create_report(run_id, dataset, path_data, result_path)


def run_pipeline_report(
    *,
    test_modes: list[str],
    eval_results: str,
    eval_results_per_drug: str,
    eval_results_per_cl: str,
    true_vs_predicted: str,
    path_data: str,
) -> None:
    """Generate HTML report from pipeline evaluation CSVs."""
    result_path = pathlib.Path(".")
    outdir_name = "report"
    create_output_directories(result_path=result_path, custom_id=outdir_name)

    ev_res = pd.read_csv(eval_results, index_col=0)
    if eval_results_per_drug == "NO_FILE":
        ev_res_per_drug = None
    else:
        ev_res_per_drug = pd.read_csv(eval_results_per_drug, index_col=0)
    if eval_results_per_cl == "NO_FILE":
        ev_res_per_cl = None
    else:
        ev_res_per_cl = pd.read_csv(eval_results_per_cl, index_col=0)
    t_vs_p = pd.read_csv(true_vs_predicted, index_col=0)

    generate_reports_for_all_test_modes(
        test_modes=test_modes,
        evaluation_results=ev_res,
        evaluation_results_per_drug=ev_res_per_drug,
        evaluation_results_per_cell_line=ev_res_per_cl,
        true_vs_pred=t_vs_p,
        run_id=outdir_name,
        path_data=path_data,
        result_path=result_path,
    )
    create_index_html(
        custom_id=outdir_name,
        test_modes=test_modes,
        prefix_results=f"{result_path}/{outdir_name}",
    )
