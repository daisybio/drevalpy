"""Generate evaluation reports after running a drug response experiment."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from upath import UPath as Path

from drevalpy.visualization.test_mode_plots import draw_test_mode_plots
from drevalpy.visualization.utils import (
    create_html,
    create_index_html,
    create_output_directories,
    draw_algorithm_plots,
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
    path_data: str | Path,
    result_path: str | Path,
) -> None:
    """Generate plots and HTML for a single test mode.

    :param test_mode: Test mode to render (for example ``"LCO"``).
    :param evaluation_results: Aggregated evaluation results.
    :param evaluation_results_per_drug: Per-drug evaluation results.
    :param evaluation_results_per_cell_line: Per-cell-line evaluation results.
    :param true_vs_pred: True versus predicted values.
    :param run_id: Unique run identifier.
    :param path_data: Path to the dataset directory.
    :param result_path: Path to the results directory.
    """
    data_dir = Path(path_data)
    results_dir = Path(result_path)

    print(f"Generating report for {test_mode} ...")
    unique_algos_ndarray = draw_test_mode_plots(
        test_mode=test_mode,
        ev_res=evaluation_results,
        ev_res_per_drug=evaluation_results_per_drug,
        ev_res_per_cell_line=evaluation_results_per_cell_line,
        custom_id=run_id,
        path_data=data_dir,
        result_path=results_dir,
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
            result_path=results_dir,
        )

    run_dir = results_dir / run_id
    all_files = []
    for entry in run_dir.rglob("*"):
        if not entry.is_file():
            continue
        if entry.name.endswith("json") or (
            entry.suffix == ".html" and entry.name not in ["index.html", "LPO.html", "LCO.html", "LDO.html"]
        ):
            all_files.append(entry.name)

    create_html(
        run_id=run_id,
        test_mode=test_mode,
        files=all_files,
        prefix_results=run_dir,
    )


def generate_reports_for_all_test_modes(
    test_modes: list[str],
    evaluation_results: pd.DataFrame,
    evaluation_results_per_drug: pd.DataFrame,
    evaluation_results_per_cell_line: pd.DataFrame,
    true_vs_pred: pd.DataFrame,
    run_id: str,
    path_data: str | Path,
    result_path: str | Path,
) -> None:
    """Generate reports for all listed test modes.

    :param test_modes: Test modes to process.
    :param evaluation_results: Aggregated evaluation results.
    :param evaluation_results_per_drug: Per-drug evaluation results.
    :param evaluation_results_per_cell_line: Per-cell-line evaluation results.
    :param true_vs_pred: True versus predicted values.
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
    path_data: str | Path = "data",
    result_path: str | Path = "results",
) -> None:
    """Render a full evaluation report pipeline.

    Parses experiment outputs, prepares aggregated tables, writes CSV summaries,
    and generates HTML plots for each test mode.

    :param run_id: Unique run identifier for locating results.
    :param dataset: Dataset name used to filter parsed results.
    :param path_data: Path to the dataset directory.
    :param result_path: Path to the experiment results directory.

    :raises AssertionError: If ``result_path/run_id`` does not exist.
    """
    data_dir = Path(path_data).resolve()
    results_dir = Path(result_path).resolve()
    run_dir = results_dir / run_id

    if not run_dir.exists():
        raise AssertionError(f"Folder {run_dir} does not exist. The pipeline has to be run first.")

    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = parse_results(path_to_results=run_dir, dataset=dataset)

    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = prep_results(evaluation_results, evaluation_results_per_drug, evaluation_results_per_cell_line, true_vs_pred)

    write_results(
        path_out=run_dir,
        eval_results=evaluation_results,
        eval_results_per_drug=evaluation_results_per_drug,
        eval_results_per_cl=evaluation_results_per_cell_line,
        t_vs_p=true_vs_pred,
    )

    create_output_directories(results_dir, run_id)
    test_modes = list(evaluation_results["test_mode"].unique())

    generate_reports_for_all_test_modes(
        test_modes=test_modes,
        evaluation_results=evaluation_results,
        evaluation_results_per_drug=evaluation_results_per_drug,
        evaluation_results_per_cell_line=evaluation_results_per_cell_line,
        true_vs_pred=true_vs_pred,
        run_id=run_id,
        path_data=data_dir,
        result_path=results_dir,
    )

    create_index_html(
        custom_id=run_id,
        test_modes=test_modes,
        prefix_results=run_dir,
    )


def run_report(
    *,
    run_id: str,
    dataset: str,
    path_data: str | Path = "data",
    result_path: str | Path = "results",
) -> None:
    """Generate HTML report from a standalone experiment run.

    :param run_id: Unique run identifier for locating results.
    :param dataset: Dataset name used to filter parsed results.
    :param path_data: Path to the dataset directory.
    :param result_path: Path to the experiment results directory.
    """
    create_report(run_id, dataset, path_data, result_path)


def run_pipeline_report(
    *,
    test_modes: list[str],
    # These stay ``str``: the pipeline passes the literal ``"NO_FILE"`` sentinel for
    # the optional CSVs, which is compared by value below.
    eval_results: str,
    eval_results_per_drug: str,
    eval_results_per_cl: str,
    true_vs_predicted: str,
    path_data: str | Path,
) -> None:
    """Generate HTML report from pipeline evaluation CSVs.

    :param test_modes: Test modes to include in the report.
    :param eval_results: Path to aggregated evaluation results CSV.
    :param eval_results_per_drug: Path to per-drug CSV, or the ``"NO_FILE"`` sentinel.
    :param eval_results_per_cl: Path to per-cell-line CSV, or the ``"NO_FILE"`` sentinel.
    :param true_vs_predicted: Path to true-versus-predicted CSV.
    :param path_data: Path to the dataset directory.
    """
    result_path = Path(".")
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
        prefix_results=result_path / outdir_name,
    )
