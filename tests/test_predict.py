"""Tests for :mod:`drevalpy._predict`: applying a fitted model to every observed pair of a dataset."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from upath import UPath

import drevalpy
from drevalpy._predict import predict_observed
from drevalpy.data import synthetic
from drevalpy.evaluation import AVAILABLE_METRICS
from drevalpy.models import DRPModel
from drevalpy.types import SplitMask
from drevalpy.types.results.run import RunResult


@pytest.fixture(scope="module")
def trained(synthetic_dataset) -> DRPModel:
    return drevalpy.train(
        "NaiveSampleMeanPredictor",
        synthetic_dataset,
        hyperparameter_tuning=False,
        response_transformation=StandardScaler(),
    )


def test_one_prediction_per_observed_pair_next_to_its_response(trained: DRPModel, synthetic_dataset) -> None:
    result = predict_observed(trained, synthetic_dataset)
    matrix = synthetic_dataset.response_matrix

    assert len(result.predictions) == int(np.isfinite(matrix).sum())
    assert np.isfinite(result.ground_truth).all()
    sample_index = {sample: i for i, sample in enumerate(synthetic_dataset.sample_ids)}
    drug_index = {drug: j for j, drug in enumerate(synthetic_dataset.drug_ids)}
    for sample, drug, truth in zip(result.sample_ids[:20], result.drug_ids[:20], result.ground_truth[:20], strict=True):
        assert matrix[sample_index[sample], drug_index[drug]] == truth
    np.testing.assert_array_equal(
        result.predictions, trained.predict(synthetic_dataset, SplitMask(np.isfinite(matrix)))
    )
    assert abs(np.nanmean(result.predictions) - np.nanmean(result.ground_truth)) < 0.5
    assert result.model_name == "NaiveSampleMeanPredictor"
    assert result.dataset_name == synthetic_dataset.name
    assert set(result.metrics) == set(AVAILABLE_METRICS)
    assert result.best_hyperparameters == trained.hyperparameters


def test_applies_to_another_dataset(trained: DRPModel) -> None:
    other = synthetic(name="OTHER", n_samples=10, seed=7)

    result = predict_observed(trained, other)

    assert result.dataset_name == "OTHER"
    assert len(result.predictions) == int(np.isfinite(other.response_matrix).sum())


def test_a_loaded_checkpoint_predicts_the_same_and_the_result_saves_as_a_run_file(
    trained: DRPModel, synthetic_dataset, tmp_path: UPath
) -> None:
    trained.save(tmp_path / "model")
    loaded = DRPModel.load(tmp_path / "model.zip")
    expected = predict_observed(trained, synthetic_dataset)

    result = predict_observed(loaded, synthetic_dataset)
    result.save(tmp_path / "predictions.npz")
    reloaded = RunResult.load(tmp_path / "predictions.npz")

    np.testing.assert_array_equal(result.predictions, expected.predictions)
    np.testing.assert_array_equal(reloaded.predictions, result.predictions)
    np.testing.assert_array_equal(reloaded.sample_ids, result.sample_ids)
    assert (reloaded.split_mode, reloaded.model_name) == ("", result.model_name)
