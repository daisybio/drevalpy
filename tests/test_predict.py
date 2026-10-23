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
    """A cheap fitted model: one mean per sample, behind a fitted response scaler."""
    return drevalpy.train(
        "NaiveSampleMeanPredictor",
        synthetic_dataset,
        hyperparameter_tuning=False,
        response_transformation=StandardScaler(),
    )


class TestPredictObserved:
    """One prediction per observed pair, next to the response it should match."""

    def test_covers_exactly_the_observed_pairs(self, trained: DRPModel, synthetic_dataset) -> None:
        result = predict_observed(trained, synthetic_dataset)

        assert len(result.predictions) == int(np.isfinite(synthetic_dataset.response_matrix).sum())
        assert np.isfinite(result.ground_truth).all()

    def test_ids_and_truth_line_up_with_the_response_matrix(self, trained: DRPModel, synthetic_dataset) -> None:
        result = predict_observed(trained, synthetic_dataset)

        sample_index = {sample: i for i, sample in enumerate(synthetic_dataset.sample_ids)}
        drug_index = {drug: j for j, drug in enumerate(synthetic_dataset.drug_ids)}
        for sample, drug, truth in zip(
            result.sample_ids[:20], result.drug_ids[:20], result.ground_truth[:20], strict=True
        ):
            assert synthetic_dataset.response_matrix[sample_index[sample], drug_index[drug]] == truth

    def test_predictions_are_the_models_in_response_units(self, trained: DRPModel, synthetic_dataset) -> None:
        result = predict_observed(trained, synthetic_dataset)

        expected = trained.predict(synthetic_dataset, SplitMask(np.isfinite(synthetic_dataset.response_matrix)))
        np.testing.assert_array_equal(result.predictions, expected)
        assert abs(np.nanmean(result.predictions) - np.nanmean(result.ground_truth)) < 0.5

    def test_names_the_model_and_dataset_and_scores_the_pairs(self, trained: DRPModel, synthetic_dataset) -> None:
        result = predict_observed(trained, synthetic_dataset)

        assert result.model_name == "NaiveSampleMeanPredictor"
        assert result.dataset_name == synthetic_dataset.name
        assert set(result.metrics) == set(AVAILABLE_METRICS)
        assert result.best_hyperparameters == trained.hyperparameters

    def test_applies_to_another_dataset(self, trained: DRPModel) -> None:
        other = synthetic(name="OTHER", n_samples=10, seed=7)

        result = predict_observed(trained, other)

        assert result.dataset_name == "OTHER"
        assert len(result.predictions) == int(np.isfinite(other.response_matrix).sum())


class TestRoundTrip:
    """A saved and loaded model predicts the same values, and the file reads back as a run file."""

    def test_a_loaded_checkpoint_predicts_the_same(self, trained: DRPModel, synthetic_dataset, tmp_path: UPath) -> None:
        trained.save(tmp_path / "model")

        loaded = DRPModel.load(tmp_path / "model.zip")

        np.testing.assert_array_equal(
            predict_observed(loaded, synthetic_dataset).predictions,
            predict_observed(trained, synthetic_dataset).predictions,
        )

    def test_the_result_saves_in_the_run_file_layout(
        self, trained: DRPModel, synthetic_dataset, tmp_path: UPath
    ) -> None:
        result = predict_observed(trained, synthetic_dataset)
        result.save(tmp_path / "predictions.npz")

        loaded = RunResult.load(tmp_path / "predictions.npz")

        np.testing.assert_array_equal(loaded.predictions, result.predictions)
        np.testing.assert_array_equal(loaded.sample_ids, result.sample_ids)
        assert loaded.split_mode == ""
        assert loaded.model_name == result.model_name
