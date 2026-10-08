"""Tests that the normalization baseline is never trained on a transformed response.

The NaiveMeanEffectsPredictor is the reference of every ``Normalized *`` metric. If it were
transformed along with the models, two runs with different ``response_transformation`` would
silently no longer be on the same scale.
"""

import os

import numpy as np
import pandas as pd
import pytest

from drevalpy.experiment import NORMALIZATION_BASELINE, drug_response_experiment
from drevalpy.models import MODEL_FACTORY
from drevalpy.response_transformation import GroupMeanCenterer

_RUN_ID = "test_normalization_baseline"


def _predictions(result_path: str, dataset_name: str, model_name: str) -> pd.DataFrame:
    """
    Collect the predictions of all folds of one model.

    :param result_path: output directory the experiment was run in
    :param dataset_name: name of the response dataset
    :param model_name: name of the model whose predictions to read
    :returns: predictions of all folds, sorted so that two runs are row-wise comparable
    """
    predictions_path = os.path.join(result_path, _RUN_ID, dataset_name, "LPO", model_name, "predictions")
    frames = []
    for file_name in sorted(os.listdir(predictions_path)):
        fold = pd.read_csv(os.path.join(predictions_path, file_name))
        fold["split"] = file_name
        frames.append(fold)
    predictions = pd.concat(frames, ignore_index=True)
    return predictions.sort_values(["split", "cell_line_name", "pubchem_id"]).reset_index(drop=True)


def _run(response_data, data_dir, path_out: str, response_transformation) -> None:
    """
    Run a minimal experiment with only the two naive baselines.

    :param response_data: response dataset, copied so that the caller's dataset stays untouched
    :param data_dir: path to the data directory
    :param path_out: output directory of this run
    :param response_transformation: transformation under test, None for the reference run
    """
    drug_response_experiment(
        models=[],
        baselines=[MODEL_FACTORY[NORMALIZATION_BASELINE], MODEL_FACTORY["NaivePredictor"]],
        response_data=response_data.copy(),
        response_transformation=response_transformation,
        run_id=_RUN_ID,
        test_mode="LPO",
        n_cv_splits=2,
        hyperparameter_tuning=False,
        path_data=str(data_dir),
        path_out=path_out,
    )


@pytest.mark.parametrize("transformation_name", ["drug_mean", "standard"])
def test_normalization_baseline_ignores_the_response_transformation(
    transformation_name, sample_dataset, data_dir, tmp_path
) -> None:
    """
    The NaiveMeanEffectsPredictor has to predict the same values with and without transformation.

    :param transformation_name: which response transformation to compare against the reference run
    :param sample_dataset: TOYv1 response data
    :param data_dir: path to the data directory
    :param tmp_path: pytest tmp_path fixture, used as output directory
    """
    from drevalpy.utils import get_response_transformation

    reference_out = str(tmp_path / "reference")
    transformed_out = str(tmp_path / transformation_name)
    _run(sample_dataset, data_dir, reference_out, None)
    _run(sample_dataset, data_dir, transformed_out, get_response_transformation(transformation_name))

    reference = _predictions(reference_out, sample_dataset.dataset_name, NORMALIZATION_BASELINE)
    transformed = _predictions(transformed_out, sample_dataset.dataset_name, NORMALIZATION_BASELINE)

    assert len(reference) == len(transformed)
    assert (reference["cell_line_name"].values == transformed["cell_line_name"].values).all()
    assert (reference["pubchem_id"].values == transformed["pubchem_id"].values).all()
    assert np.allclose(reference["response"], transformed["response"])
    assert np.allclose(reference["predictions"], transformed["predictions"])


def test_the_other_models_are_still_transformed(sample_dataset, data_dir, tmp_path) -> None:
    """
    The exemption has to be limited to the normalization baseline, drug_mean still has to reach the rest.

    :param sample_dataset: TOYv1 response data
    :param data_dir: path to the data directory
    :param tmp_path: pytest tmp_path fixture, used as output directory
    """
    reference_out = str(tmp_path / "reference")
    transformed_out = str(tmp_path / "drug_mean")
    _run(sample_dataset, data_dir, reference_out, None)
    _run(sample_dataset, data_dir, transformed_out, GroupMeanCenterer())

    reference = _predictions(reference_out, sample_dataset.dataset_name, "NaivePredictor")
    transformed = _predictions(transformed_out, sample_dataset.dataset_name, "NaivePredictor")

    # Without transformation NaivePredictor predicts the training mean for every row, with
    # drug_mean it predicts the per-drug mean of the training fold.
    assert reference["predictions"].nunique() < transformed["predictions"].nunique()
    assert not np.allclose(reference["predictions"], transformed["predictions"])
