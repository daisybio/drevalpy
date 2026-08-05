"""CLI/experiment paths must not emit legacy factory or view FutureWarnings."""

from __future__ import annotations

import os
import pathlib
import warnings

import yaml
from typer.testing import CliRunner

from drevalpy._deprecations import reset_deprecation_warnings
from drevalpy.cli.main import app
from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models import construct_model
from drevalpy.models._model_lookup import get_model_class, known_model_names
from drevalpy.utils import check_arguments
from drevalpy.utils.pickle_io import dump_trusted_pickle

runner = CliRunner()


def _dreval_future_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [
        w
        for w in caught
        if issubclass(w.category, FutureWarning)
        and (
            "MODEL_FACTORY" in str(w.message)
            or "MULTI_DRUG_MODEL_FACTORY" in str(w.message)
            or "SINGLE_DRUG_MODEL_FACTORY" in str(w.message)
            or "Legacy cell_line_views/drug_views" in str(w.message)
        )
    ]


def test_check_arguments_and_model_lookup_do_not_warn(tmp_path: pathlib.Path) -> None:
    reset_deprecation_warnings()

    class _Args:
        models = ["ElasticNet"]
        baselines = ["NaivePredictor"]
        test_mode = ["LPO"]
        dataset_name = "GDSC1"
        path_data = str(tmp_path / "drevalpy-data")
        no_refitting = True
        curve_curator_cores = 1
        cross_study_datasets: list[str] = []
        n_cv_splits = 2
        custom_splitter_path = None
        custom_split_name = None
        randomization_mode = ["None"]
        randomization_type = "permutation"
        n_trials_robustness = 0
        response_transformation = "None"
        measure = "LN_IC50"
        optim_metric = "Pearson"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert "ElasticNet" in known_model_names(include_external=False)
        assert get_model_class("ElasticNet").get_model_name() == "ElasticNet"
        assert construct_model("ElasticNet").get_model_name() == "ElasticNet"
        check_arguments(_Args())

    assert _dreval_future_warnings(caught) == []


def test_test_cv_with_modern_hpams_does_not_warn_on_legacy_views(
    data_dir: pathlib.Path,
    sample_dataset: DrugResponseDataset,
    tmp_path: pathlib.Path,
) -> None:
    """Modern best-hpam YAML without view keys should not emit view FutureWarnings.

    :param data_dir: Path to the drevalpy data directory.
    :param sample_dataset: TOYv1 dataset fixture used to build a CV split.
    :param tmp_path: Temporary directory for split and HPAM artifacts.
    """
    reset_deprecation_warnings()
    cv_splits = sample_dataset.split_dataset(n_cv_splits=5, mode="LCO", random_state=42)
    split = cv_splits[0]

    split_path = tmp_path / "split_0.pkl"
    dump_trusted_pickle(split, split_path)

    hpam_path = tmp_path / "best_hpam_combi_split_0.yaml"
    with open(hpam_path, "w") as fh:
        yaml.dump(
            {
                "ElasticNet_split_0": {
                    "best_hpam_combi": {
                        "alpha": 0.1,
                        "l1_ratio": 0.5,
                    }
                }
            },
            fh,
        )

    prev_dir = os.getcwd()
    try:
        os.chdir(tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = runner.invoke(
                app,
                [
                    "test-cv",
                    "--mode",
                    "full",
                    "--model_name",
                    "ElasticNet",
                    "--split_id",
                    "split_0",
                    "--split_dataset_path",
                    str(split_path),
                    "--hyperparameters_path",
                    str(hpam_path),
                    "--path_data",
                    str(data_dir),
                    "--test_mode",
                    "LCO",
                ],
            )
    finally:
        os.chdir(prev_dir)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}:\n{result.output}"
    assert _dreval_future_warnings(caught) == []
