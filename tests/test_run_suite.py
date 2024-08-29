"""
Tests whether the main function of the package runs without errors and produces the expected output.
"""

import os
from argparse import Namespace
import tempfile
import pytest

from drevalpy.utils import main
from drevalpy.visualization.utils import parse_results, prep_results


@pytest.mark.parametrize(
    "args",
    [
        {
            "run_id": "test_run",
            "dataset_name": "Toy_Data",
            "models": ["ElasticNet"],
            "baselines": ["NaiveDrugMeanPredictor"],
            "test_mode": ["LPO"],
            "randomization_mode": ["SVRC"],
            "randomization_type": "permutation",
            "n_trials_robustness": 2,
            "cross_study_datasets": [],
            "curve_curator": False,
            "overwrite": False,
            "optim_metric": "RMSE",
            "n_cv_splits": 5,
            "response_transformation": "None",
            "multiprocessing": False,
            "path_data": "../data",
        }
    ],
)
def test_run_suite(args):
    """
    Tests run_suite.py, i.e., all functionality of the main package
    :param args:
    :return:
    """
    temp_dir = tempfile.TemporaryDirectory()
    args["path_out"] = temp_dir.name
    args = Namespace(**args)
    main(args)
    assert os.listdir(temp_dir.name) == ["test_run"]
    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = parse_results(path_to_results=f"{temp_dir.name}/{args.run_id}")

    (
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    ) = prep_results(
        evaluation_results,
        evaluation_results_per_drug,
        evaluation_results_per_cell_line,
        true_vs_pred,
    )

    assert len(evaluation_results.columns) == 22
    assert len(evaluation_results_per_drug.columns) == 15
    assert len(evaluation_results_per_cell_line.columns) == 15
    assert len(true_vs_pred.columns) == 12

    assert all(model in evaluation_results.algorithm.unique() for model in args.models)
    assert all(
        baseline in evaluation_results.algorithm.unique() for baseline in args.baselines
    )
    assert "predictions" in evaluation_results.rand_setting.unique()
    if len(args.randomization_mode) > 0:
        for rand_setting in args.randomization_mode:
            assert any(
                setting.startswith(f"randomize-{rand_setting}")
                for setting in evaluation_results.rand_setting.unique()
            )
    if args.n_trials_robustness > 0:
        assert any(
            setting.startswith(f"robustness-{args.n_trials_robustness}")
            for setting in evaluation_results.rand_setting.unique()
        )
    assert all(
        test_mode in evaluation_results.LPO_LCO_LDO.unique()
        for test_mode in args.test_mode
    )
    assert evaluation_results.CV_split.astype(int).max() == (args.n_cv_splits - 1)
    assert evaluation_results.Pearson.astype(float).max() > 0.5
