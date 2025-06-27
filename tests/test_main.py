"""Test suite for the main functionality of drevalpy."""

import os
import pathlib
import tempfile
from argparse import Namespace

import pytest

from drevalpy.utils import check_arguments, get_parser, main
from drevalpy.visualization.create_report import create_report
from drevalpy.visualization.utils import (
    create_output_directories,
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
            "no_refitting": False,
            "curve_curator_cores": 1,
            "measure": "LN_IC50",
            "overwrite": False,
            "optim_metric": "RMSE",
            "n_cv_splits": 2,
            "response_transformation": "standard",
            "multiprocessing": False,
            "path_data": "../data",
            "model_checkpoint_dir": "TEMPORARY",
            "no_hyperparameter_tuning": True,
            "final_model_on_full_data": True,
        }
    ],
)
def test_drevalpy_main(args):
    """
    Tests drevalpy, i.e., all functionality of the main experiment and report.

    :param args: arguments for the main function
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        args["path_out"] = temp_dir
        args = Namespace(**args)
        get_parser()
        check_arguments(args)

        try:
            main(args)
        except Exception as e:
            pytest.fail(f"Main function failed: {e}")

        # Check output directory contains the run_id folder
        assert args.run_id in os.listdir(temp_dir)

        # Run report generation on the output of the main run
        try:
            create_report(args.run_id, args.dataset_name, args.path_data, temp_dir)
        except Exception as e:
            pytest.fail(f"Report generation failed: {e}")

        result_path = pathlib.Path(temp_dir).resolve()
        path_data = pathlib.Path(args.path_data).resolve()

        # Parse and prep results
        (
            evaluation_results,
            evaluation_results_per_drug,
            evaluation_results_per_cell_line,
            true_vs_pred,
        ) = parse_results(path_to_results=f"{result_path}/{args.run_id}", dataset=args.dataset_name)

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

        # Basic structural assertions
        expected_eval_cols = 15
        expected_tvp_cols = 11
        assert len(evaluation_results.columns) == expected_eval_cols
        assert len(evaluation_results_per_drug.columns) == expected_eval_cols
        assert len(evaluation_results_per_cell_line.columns) == expected_eval_cols
        assert len(true_vs_pred.columns) == expected_tvp_cols

        # Check models and baselines present in evaluation results
        assert all(model in evaluation_results.algorithm.unique() for model in args.models)
        assert all(baseline in evaluation_results.algorithm.unique() for baseline in args.baselines)
        assert "predictions" in evaluation_results.rand_setting.unique()

        # Check randomization modes in rand_setting
        if args.randomization_mode:
            for rand_setting in args.randomization_mode:
                assert any(
                    setting.startswith(f"randomize-{rand_setting}")
                    for setting in evaluation_results.rand_setting.unique()
                )

        # Check robustness trials presence
        if args.n_trials_robustness > 0:
            assert any(
                setting.startswith(f"robustness-{args.n_trials_robustness}")
                for setting in evaluation_results.rand_setting.unique()
            )

        # Check test modes and CV splits
        assert all(test_mode in evaluation_results.test_mode.unique() for test_mode in args.test_mode)
        assert evaluation_results.CV_split.astype(int).max() == (args.n_cv_splits - 1)

        # Check some metric threshold
        assert evaluation_results.Pearson.astype(float).max() > 0.5

        # Verify output directories exist (from report generation)
        create_output_directories(result_path, args.run_id)
