"""Tests for the model-level result that aggregates runs across folds."""

from __future__ import annotations

import json

import numpy as np
import pytest
from upath import UPath

from drevalpy.types.results.model import ModelResult
from drevalpy.types.results.trial import TrialResult
from tests.synthetic import make_model_result, make_run_result


class TestFoldCount:
    def test_counts_the_contained_runs(self) -> None:
        assert make_model_result(n_folds=4).n_folds == 4

    def test_is_zero_without_runs(self) -> None:
        assert ModelResult(model_name="ElasticNet", dataset_name="SyntheticDataset").n_folds == 0


class TestAggregateMetrics:
    def test_is_empty_without_runs(self) -> None:
        result = ModelResult(model_name="ElasticNet", dataset_name="SyntheticDataset")

        assert result.aggregate_metrics == {}

    def test_reports_mean_and_std_across_folds(self) -> None:
        result = ModelResult(
            model_name="ElasticNet",
            dataset_name="SyntheticDataset",
            runs=[
                make_run_result(fold_index=0, metrics={"MSE": 0.2}),
                make_run_result(fold_index=1, metrics={"MSE": 0.4}),
            ],
        )

        assert result.aggregate_metrics["MSE"]["mean"] == pytest.approx(0.3)
        assert result.aggregate_metrics["MSE"]["std"] == pytest.approx(0.1)

    def test_std_is_zero_for_a_single_fold(self) -> None:
        result = ModelResult(
            model_name="ElasticNet",
            dataset_name="SyntheticDataset",
            runs=[make_run_result(metrics={"MSE": 0.7})],
        )

        assert result.aggregate_metrics["MSE"] == {"mean": pytest.approx(0.7), "std": pytest.approx(0.0)}

    def test_unions_metric_keys_across_folds(self) -> None:
        result = ModelResult(
            model_name="ElasticNet",
            dataset_name="SyntheticDataset",
            runs=[
                make_run_result(fold_index=0, metrics={"MSE": 0.2}),
                make_run_result(fold_index=1, metrics={"MSE": 0.4, "Pearson": 0.9}),
            ],
        )

        assert set(result.aggregate_metrics) == {"MSE", "Pearson"}
        assert result.aggregate_metrics["Pearson"]["mean"] == pytest.approx(0.9)

    def test_covers_every_metric_the_builder_emits(self) -> None:
        result = make_model_result(n_folds=3)

        assert set(result.aggregate_metrics) == set(result.runs[0].metrics)


class TestPersistence:
    def test_save_creates_missing_parent_directories(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "nested" / "ElasticNet"

        make_model_result(n_folds=1).save(directory)

        assert (directory / "metadata.json").is_file()

    def test_save_writes_one_npz_per_fold(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "ElasticNet"

        make_model_result(n_folds=3).save(directory)

        assert sorted(p.name for p in directory.glob("fold_*.npz")) == [
            "fold_0.npz",
            "fold_1.npz",
            "fold_2.npz",
        ]

    def test_save_records_aggregates_in_metadata(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "ElasticNet"
        result = make_model_result(n_folds=2)

        result.save(directory)
        meta = json.loads((directory / "metadata.json").read_text())

        assert meta["model_name"] == "ElasticNet"
        assert meta["dataset_name"] == "SyntheticDataset"
        assert meta["n_folds"] == 2
        assert meta["aggregate_metrics"]["MSE"]["mean"] == pytest.approx(result.aggregate_metrics["MSE"]["mean"])

    def test_round_trip_preserves_identity_and_folds(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "RandomForest"
        result = make_model_result(model_name="RandomForest", n_folds=3)

        result.save(directory)
        loaded = ModelResult.load(directory)

        assert loaded.model_name == "RandomForest"
        assert loaded.dataset_name == "SyntheticDataset"
        assert loaded.n_folds == 3

    def test_round_trip_preserves_fold_order(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "ElasticNet"
        result = make_model_result(n_folds=3)

        result.save(directory)
        loaded = ModelResult.load(directory)

        assert [r.fold_index for r in loaded.runs] == [0, 1, 2]
        np.testing.assert_allclose(loaded.runs[1].predictions, result.runs[1].predictions)

    def test_round_trip_of_an_empty_result_yields_no_runs(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "ElasticNet"
        ModelResult(model_name="ElasticNet", dataset_name="SyntheticDataset").save(directory)

        assert ModelResult.load(directory).runs == []

    def test_accepts_a_plain_string_directory(self, tmp_path) -> None:
        directory = str(UPath(tmp_path) / "ElasticNet")

        make_model_result(n_folds=1).save(directory)

        assert ModelResult.load(directory).n_folds == 1


class TestTrialSkipping:
    """``with_trials`` is forwarded to every fold so the report can skip trial arrays."""

    @staticmethod
    def _with_trials(directory: UPath) -> None:
        result = make_model_result(n_folds=2)
        for run in result.runs:
            run.trials = [
                TrialResult(
                    hyperparameters={"alpha": 0.1},
                    metrics={"MSE": 0.3},
                    optimization_metric="MSE",
                    predictions=np.zeros(4),
                )
            ]
        result.save(directory)

    def test_trials_are_loaded_by_default(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "ElasticNet"
        self._with_trials(directory)

        assert all(run.trials for run in ModelResult.load(directory).runs)

    def test_trials_can_be_skipped_for_every_fold(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "ElasticNet"
        self._with_trials(directory)

        assert all(run.trials is None for run in ModelResult.load(directory, with_trials=False).runs)

    def test_skipping_trials_keeps_the_fold_count(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "ElasticNet"
        self._with_trials(directory)

        assert ModelResult.load(directory, with_trials=False).n_folds == 2


class TestRepr:
    def test_reports_identity_and_fold_count(self) -> None:
        text = repr(make_model_result(model_name="RandomForest", n_folds=2))

        assert text.startswith("ModelResult")
        assert "Model: RandomForest" in text
        assert "Dataset: SyntheticDataset" in text
        assert "Folds: 2" in text

    def test_formats_metrics_as_mean_plus_minus_std(self) -> None:
        result = ModelResult(
            model_name="ElasticNet",
            dataset_name="SyntheticDataset",
            runs=[
                make_run_result(fold_index=0, metrics={"MSE": 0.2}),
                make_run_result(fold_index=1, metrics={"MSE": 0.4}),
            ],
        )

        assert "        MSE: 0.3000 +/- 0.1000" in repr(result).splitlines()

    def test_omits_the_metric_block_without_runs(self) -> None:
        result = ModelResult(model_name="ElasticNet", dataset_name="SyntheticDataset")

        assert "Metrics" not in repr(result)
