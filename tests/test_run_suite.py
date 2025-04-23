"""Tests whether the main function of the package runs without errors and produces the expected output."""

import os
import pathlib
import tempfile
from argparse import Namespace

import pytest

from drevalpy.utils import check_arguments, get_parser, main
from drevalpy.visualization.utils import (
    create_html,
    create_index_html,
    create_output_directories,
    draw_algorithm_plots,
    draw_setting_plots,
    parse_results,
    prep_results,
)


@pytest.mark.parametrize(
    "args",
    [
        {
            "run_id": "test_run",
            "dataset_name": "TOYv1",
            "models": ["ElasticNet"],
            "baselines": ["NaiveMeanEffectsPredictor", "NaivePredictor"],
            "test_mode": ["LPO"],
            "randomization_mode": ["SVRC"],
            "randomization_type": "permutation",
            "n_trials_robustness": 2,
            "cross_study_datasets": ["TOYv2"],
            "curve_curator": True,
            "curve_curator_cores": 1,
            "measure": "LN_IC50",
            "overwrite": False,
            "optim_metric": "RMSE",
            "n_cv_splits": 2,
            "response_transformation": "None",
            "multiprocessing": False,
            "path_data": "../data",
            "model_checkpoint_dir": "TEMPORARY",
        }
    ],
)
def test_run_suite(args):
    """
    Tests run_suite.py, i.e., all functionality of the main package.

    :param args: arguments for the main function
    """
    temp_dir = tempfile.TemporaryDirectory()
    args["path_out"] = temp_dir.name
    args = Namespace(**args)
    get_parser()
    check_arguments(args)
    main(args, debug_mode=True)
    assert os.listdir(temp_dir.name) == ["test_run"]
    result_path = pathlib.Path(temp_dir.name).resolve()
    path_data = pathlib.Path(args.path_data).resolve()

    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = parse_results(path_to_results=f"{result_path}/{args.run_id}", dataset="TOYv1")

    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = prep_results(
        eval_results=evaluation_results,
        eval_results_per_drug=evaluation_results_per_drug,
        eval_results_per_cell_line=evaluation_results_per_cell_line,
        t_vs_p=true_vs_pred,
        path_data=path_data,
    )
    assert len(evaluation_results.columns) == 15
    assert len(evaluation_results_per_drug.columns) == 15
    assert len(evaluation_results_per_cell_line.columns) == 15
    assert len(true_vs_pred.columns) == 11

    assert all(model in evaluation_results.algorithm.unique() for model in args.models)
    assert all(baseline in evaluation_results.algorithm.unique() for baseline in args.baselines)
    assert "predictions" in evaluation_results.rand_setting.unique()
    if len(args.randomization_mode) > 0:
        for rand_setting in args.randomization_mode:
            assert any(
                setting.startswith(f"randomize-{rand_setting}") for setting in evaluation_results.rand_setting.unique()
            )
    if args.n_trials_robustness > 0:
        assert any(
            setting.startswith(f"robustness-{args.n_trials_robustness}")
            for setting in evaluation_results.rand_setting.unique()
        )
    assert all(test_mode in evaluation_results.LPO_LCO_LDO.unique() for test_mode in args.test_mode)
    assert evaluation_results.CV_split.astype(int).max() == (args.n_cv_splits - 1)
    assert evaluation_results.Pearson.astype(float).max() > 0.5

    create_output_directories(result_path, args.run_id)
    setting = args.test_mode[0]
    unique_algos = draw_setting_plots(
        lpo_lco_ldo=setting,
        ev_res=evaluation_results,
        ev_res_per_drug=evaluation_results_per_drug,
        ev_res_per_cell_line=evaluation_results_per_cell_line,
        custom_id=args.run_id,
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
            lpo_lco_ldo=setting,
            custom_id=args.run_id,
            result_path=result_path,
        )
    # get all html files from {result_path}/{run_id}
    all_files: list[str] = []
    for _, _, files in os.walk(f"{result_path}/{args.run_id}"):  # type: ignore[assignment]
        for file in files:
            if file.endswith("json") or (
                file.endswith(".html") and file not in ["index.html", "LPO.html", "LCO.html", "LDO.html"]
            ):
                all_files.append(file)
    # PIPELINE: WRITE_HTML
    create_html(
        run_id=args.run_id,
        lpo_lco_ldo=setting,
        files=all_files,
        prefix_results=f"{result_path}/{args.run_id}",
        test_mode=setting,
    )
    # PIPELINE: WRITE_INDEX
    create_index_html(
        custom_id=args.run_id,
        test_modes=args.test_mode,
        prefix_results=f"{result_path}/{args.run_id}",
    )
