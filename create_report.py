"""Renders the evaluation results into an HTML report with various plots and tables."""

import argparse
import os
import pathlib

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reports from evaluation results")
    parser.add_argument("--run_id", required=True, help="Run ID for the current execution")
    parser.add_argument("--dataset", required=True, help="Dataset name for which to render the result file")
    parser.add_argument("--path_data", required=False, help="Path to the data, default is ./data")
    parser.add_argument("--result_path", required=False, help="Path to the results, default is ./results")
    args = parser.parse_args()
    run_id = args.run_id
    dataset = args.dataset
    path_to_data = args.path_data if args.path_data is not None else "data"
    path_data = pathlib.Path(path_to_data).resolve()
    result_path = args.result_path if args.result_path is not None else "results"
    result_path = pathlib.Path(result_path).resolve()
    # assert that the run_id folder exists
    if not os.path.exists(f"{result_path}/{run_id}"):
        raise AssertionError(f"Folder {result_path}/{run_id} does not exist. The pipeline has to be run first.")

    # not part of pipeline
    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = parse_results(path_to_results=f"{result_path}/{run_id}", dataset=dataset)

    # part of pipeline: EVALUATE_FINAL, COLLECT_RESULTS
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
    # Start loop over all test_modes
    test_modes = evaluation_results["test_mode"].unique()

    for test_mode in test_modes:
        print(f"Generating report for {test_mode} ...")
        unique_algos = draw_test_mode_plots(
            test_mode=test_mode,
            ev_res=evaluation_results,
            ev_res_per_drug=evaluation_results_per_drug,
            ev_res_per_cell_line=evaluation_results_per_cell_line,
            custom_id=run_id,
            path_data=path_data,
            result_path=result_path,
        )
        # draw figures for each algorithm with all randomizations etc
        unique_algos = set(unique_algos) - {
            "NaiveMeanEffectsPredictor",
            "NaivePredictor",
            "NaiveCellLineMeansPredictor",
            "NaiveDrugMeanPredictor",
        }
        for algorithm in unique_algos:
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
        # get all html files from {result_path}/{run_id}
        all_files: list[str] = []
        for _, _, files in os.walk(f"{result_path}/{run_id}"):  # type: ignore[assignment]
            for file in files:
                if file.endswith("json") or (
                    file.endswith(".html") and file not in ["index.html", "LPO.html", "LCO.html", "LDO.html"]
                ):
                    all_files.append(file)
        # PIPELINE: WRITE_HTML
        create_html(
            run_id=run_id,
            test_mode=test_mode,
            files=all_files,
            prefix_results=f"{result_path}/{run_id}",
        )
    # PIPELINE: WRITE_INDEX
    create_index_html(
        custom_id=run_id,
        test_modes=test_modes,
        prefix_results=f"{result_path}/{run_id}",
    )
