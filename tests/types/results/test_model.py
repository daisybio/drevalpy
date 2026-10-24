"""Tests for the model-level result that aggregates runs across folds."""

from __future__ import annotations

import json

import numpy as np
import pytest
from upath import UPath

from drevalpy.types.results.model import ModelResult
from drevalpy.types.results.trial import TrialResult
from tests.synthetic import make_model_result, make_run_result


def _two_fold_mse() -> ModelResult:
    return ModelResult(
        "ElasticNet",
        "SyntheticDataset",
        [make_run_result(fold_index=0, metrics={"MSE": 0.2}), make_run_result(fold_index=1, metrics={"MSE": 0.4})],
    )


class TestFoldCount:
    def test_counts_runs(self) -> None:
        assert make_model_result(n_folds=4).n_folds == 4
        assert ModelResult("ElasticNet", "SyntheticDataset").n_folds == 0

    def test_distinct_folds_are_counted_separately_from_diagnostic_runs(self) -> None:
        ordinary = make_run_result(fold_index=0)
        robustness = make_run_result(fold_index=0, fold_metadata={"robustness_trial": 0})
        randomized = make_run_result(fold_index=0, randomization=("SVRC", "gene_expression"))
        result = ModelResult("ElasticNet", "SyntheticDataset", [ordinary, robustness, randomized])

        assert result.n_folds == 1
        assert result.n_runs == 3
        assert result.select_runs() == [ordinary]
        assert result.select_runs(randomization="all", robustness="all") == [ordinary, robustness, randomized]


class TestAggregateGroupMetrics:
    def test_is_empty_without_runs(self) -> None:
        assert ModelResult("ElasticNet", "SyntheticDataset").aggregate_group_metrics("drug") == {}

    def test_reports_a_mean_and_a_group_count_per_metric(self) -> None:
        stats = make_model_result(n_folds=2, n_pairs=40).aggregate_group_metrics("drug")["Pearson"]

        assert set(stats) == {"mean", "n_groups"}
        assert stats["n_groups"] == 4
        assert make_model_result(n_folds=1, n_pairs=40).aggregate_group_metrics("sample")["MSE"]["n_groups"] == 5

    def test_covers_the_metrics_the_runs_carry(self) -> None:
        result = ModelResult(
            "ElasticNet", "SyntheticDataset", [make_run_result(metrics={"MSE": 0.2, "Not a metric": 1.0}, n_pairs=40)]
        )

        assert list(result.aggregate_group_metrics("drug")) == ["MSE"]

    def test_an_unknown_grouping_is_rejected_even_without_metrics(self) -> None:
        result = ModelResult("ElasticNet", "SyntheticDataset", [make_run_result(metrics={})])

        with pytest.raises(ValueError, match="Unknown grouping"):
            result.aggregate_group_metrics("tissue")  # type: ignore[arg-type]


class TestAggregateMetrics:
    def test_is_empty_without_runs(self) -> None:
        assert ModelResult("ElasticNet", "SyntheticDataset").aggregate_metrics == {}

    def test_reports_mean_and_std_across_folds(self) -> None:
        assert _two_fold_mse().aggregate_metrics["MSE"]["mean"] == pytest.approx(0.3)
        assert _two_fold_mse().aggregate_metrics["MSE"]["std"] == pytest.approx(0.1)

    def test_std_is_zero_for_a_single_fold(self) -> None:
        result = ModelResult("ElasticNet", "SyntheticDataset", [make_run_result(metrics={"MSE": 0.7})])

        assert result.aggregate_metrics["MSE"] == {"mean": pytest.approx(0.7), "std": pytest.approx(0.0)}

    def test_unions_metric_keys_across_folds(self) -> None:
        result = ModelResult(
            "ElasticNet",
            "SyntheticDataset",
            [
                make_run_result(fold_index=0, metrics={"MSE": 0.2}),
                make_run_result(fold_index=1, metrics={"MSE": 0.4, "Pearson": 0.9}),
            ],
        )

        assert set(result.aggregate_metrics) == {"MSE", "Pearson"}
        assert result.aggregate_metrics["Pearson"]["mean"] == pytest.approx(0.9)


class TestPersistence:
    def test_save_writes_folds_and_metadata(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "nested" / "ElasticNet"
        result = make_model_result(n_folds=3)

        result.save(directory)
        meta = json.loads((directory / "metadata.json").read_text())

        assert sorted(p.name for p in directory.glob("fold_*.npz")) == ["fold_0.npz", "fold_1.npz", "fold_2.npz"]
        assert (meta["model_name"], meta["dataset_name"]) == ("ElasticNet", "SyntheticDataset")
        assert (meta["n_folds"], meta["n_runs"]) == (3, 3)
        assert meta["aggregate_metrics"]["MSE"]["mean"] == pytest.approx(result.aggregate_metrics["MSE"]["mean"])

    def test_round_trip_preserves_identity_and_fold_order(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "RandomForest"
        result = make_model_result(model_name="RandomForest", n_folds=3)

        result.save(directory)
        loaded = ModelResult.load(directory)

        assert (loaded.model_name, loaded.dataset_name, loaded.n_folds) == ("RandomForest", "SyntheticDataset", 3)
        assert [r.fold_index for r in loaded.runs] == [0, 1, 2]
        np.testing.assert_allclose(loaded.runs[1].predictions, result.runs[1].predictions)

    def test_double_digit_execution_order_and_resave_do_not_revive_old_runs(self, tmp_path) -> None:
        result = make_model_result(n_folds=12)
        result.save(tmp_path)
        assert [r.fold_index for r in ModelResult.load(tmp_path).runs] == list(range(12))
        result.runs = result.runs[:2]
        result.save(tmp_path)
        assert [r.fold_index for r in ModelResult.load(tmp_path).runs] == [0, 1]

    def test_empty_result_round_trips_and_string_directory_is_accepted(self, tmp_path) -> None:
        empty = UPath(tmp_path) / "empty"
        ModelResult("ElasticNet", "SyntheticDataset").save(empty)
        assert ModelResult.load(empty).runs == []

        directory = str(UPath(tmp_path) / "ElasticNet")
        make_model_result(n_folds=1).save(directory)
        assert ModelResult.load(directory).n_folds == 1

    def test_trials_are_forwarded_to_every_fold(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "ElasticNet"
        result = make_model_result(n_folds=2)
        for run in result.runs:
            run.trials = [TrialResult({"alpha": 0.1}, {"MSE": 0.3}, "MSE", np.zeros(4))]
        result.save(directory)

        assert all(run.trials for run in ModelResult.load(directory).runs)
        skipped = ModelResult.load(directory, with_trials=False)
        assert all(run.trials is None for run in skipped.runs)
        assert skipped.n_folds == 2


class TestRepr:
    def test_reports_identity_fold_count_and_metrics(self) -> None:
        text = repr(make_model_result(model_name="RandomForest", n_folds=2))

        assert text.startswith("ModelResult")
        assert "Model: RandomForest" in text
        assert "Folds: 2" in text
        assert "        MSE: 0.3000 +/- 0.1000" in repr(_two_fold_mse()).splitlines()

    def test_omits_the_metric_block_without_runs(self) -> None:
        assert "Metrics" not in repr(ModelResult("ElasticNet", "SyntheticDataset"))
