"""Test suite for the main functionality of drevalpy."""

import os
import pathlib
import tempfile
from argparse import Namespace

import pytest

from drevalpy.utils import main


@pytest.mark.parametrize(
    "args",
    [
        {
            "run_id": "test_run",
            "dataset_name": "TOYv2",
            "models": ["NaiveMeanEffectsPredictor"],
            "baselines": None,
            "test_mode": ["LPO"],
            "overwrite": False,
            "optim_metric": "RMSE",
            "n_cv_splits": 2,
            "response_transformation": "standard",
            "model_checkpoint_dir": None,
            "no_hyperparameter_tuning": True,
            "wandb_project": None,
        }
    ],
)
def test_drevalpy_main(args, data_dir):
    """Test the MuData-based pipeline runs to completion.

    :param args: arguments for the main function
    :param data_dir: path to the data directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        args["path_out"] = temp_dir
        args = Namespace(**args)

        try:
            main(args)
        except Exception as e:
            pytest.fail(f"Main function failed: {e}")

        assert args.run_id in os.listdir(temp_dir)

        run_dir = pathlib.Path(temp_dir) / args.run_id
        assert run_dir.exists()
