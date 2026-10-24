"""Tests for :mod:`drevalpy.visualization.plots._leaderboard_table` (the 1.6.1 ``load_results`` table)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from drevalpy.evaluation import score_predictions
from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.types.results.run import RunResult
from drevalpy.visualization.plots._leaderboard_table import build_leaderboard_df
from tests.synthetic import REFERENCE_MODEL, make_run_result

COLUMNS = [
    "algorithm",
    "PCC",
    "PCC_std",
    "RMSE",
    "RMSE_std",
    "Pearson",
    "Pearson_std",
    "Pearson_per_drug",
    "Pearson_per_drug_std",
    "is_baseline",
]


def _run(model: str, fold: int, seed: int, *, noise: float = 0.5) -> RunResult:
    rng = np.random.default_rng(seed)
    drugs = np.repeat([f"D{i}" for i in range(4)], 6).astype(object)
    samples = np.tile([f"S{i}" for i in range(6)], 4).astype(object)
    truth = rng.normal(size=24) + np.repeat([0.0, 2.0, 4.0, 6.0], 6)
    predictions = truth + rng.normal(scale=noise, size=24)
    return RunResult(
        model_name=model,
        dataset_name="D",
        fold_index=fold,
        predictions=predictions,
        ground_truth=truth,
        sample_ids=samples,
        drug_ids=drugs,
        split_mode="LSO",
        fold_id=f"fold_{fold}",
        metrics=score_predictions(predictions, truth, empty_as_dict=True),
    )


def _per_drug_pearson(run: RunResult) -> float:
    return float(
        np.mean(
            [
                np.corrcoef(run.predictions[run.drug_ids == drug], run.ground_truth[run.drug_ids == drug])[0, 1]
                for drug in np.unique(run.drug_ids)
            ]
        )
    )


@pytest.fixture
def experiment() -> ExperimentResult:
    runs = [
        _run(name, fold, seed=10 * index + fold, noise=noise)
        for index, (name, noise) in enumerate([(REFERENCE_MODEL, 2.0), ("ElasticNet", 1.0), ("RandomForest", 0.4)])
        for fold in range(3)
    ]
    return ExperimentResult(runs)


def _runs_of(experiment: ExperimentResult, name: str) -> list[RunResult]:
    return next(m for m in experiment.models if m.model_name == name).runs


class TestUnnormalized:
    def test_schema_ordering_and_baseline_flags(self, experiment):
        df = build_leaderboard_df(experiment)

        assert list(df.columns) == COLUMNS
        assert len(df) == len(experiment.models)
        assert df["PCC"].is_monotonic_decreasing
        assert df["Pearson"].tolist() == df["PCC"].tolist()
        flags = df.set_index("algorithm")["is_baseline"]
        assert bool(flags[REFERENCE_MODEL]) is True
        assert bool(flags["ElasticNet"]) is False

    def test_means_and_sample_std_are_taken_over_folds(self, experiment):
        row = build_leaderboard_df(experiment).set_index("algorithm").loc["ElasticNet"]
        runs = _runs_of(experiment, "ElasticNet")
        pearson = [r.metrics["Pearson"] for r in runs]

        assert row["PCC"] == pytest.approx(np.mean(pearson))
        assert row["PCC_std"] == pytest.approx(np.std(pearson, ddof=1))
        assert row["RMSE"] == pytest.approx(np.mean([r.metrics["RMSE"] for r in runs]))

    def test_per_drug_pearson_averages_the_drugs_then_the_folds(self, experiment):
        row = build_leaderboard_df(experiment).set_index("algorithm").loc["RandomForest"]
        per_fold = [_per_drug_pearson(r) for r in _runs_of(experiment, "RandomForest")]

        assert row["Pearson_per_drug"] == pytest.approx(np.mean(per_fold))
        assert row["Pearson_per_drug_std"] == pytest.approx(np.std(per_fold, ddof=1))

    def test_single_fold_models_get_a_zero_standard_deviation(self):
        df = build_leaderboard_df(ExperimentResult([_run("Solo", 0, seed=1)]))

        assert df.filter(like="_std").iloc[0].tolist() == [0.0] * 4

    def test_a_constant_prediction_within_every_drug_has_no_per_drug_pearson(self):
        run = _run("Constant", 0, seed=2)
        run.predictions = np.repeat(run.ground_truth[::6], 6)

        df = build_leaderboard_df(ExperimentResult([run]))

        assert np.isnan(df["Pearson_per_drug"].iloc[0])
        assert df["Pearson_per_drug_std"].iloc[0] == 0.0

    def test_randomized_runs_are_excluded_and_all_randomized_gives_the_empty_schema(self):
        randomized = make_run_result(model_name="RandomForest", randomization=("gene_expression", "permutation"))
        result = ExperimentResult(
            [
                make_run_result(model_name="ElasticNet", fold_index=0),
                make_run_result(model_name="ElasticNet", fold_index=1),
                randomized,
            ]
        )

        assert build_leaderboard_df(result)["algorithm"].tolist() == ["ElasticNet"]
        empty = build_leaderboard_df(ExperimentResult([randomized]))
        assert empty.empty
        assert list(empty.columns) == COLUMNS

    def test_reads_the_plain_pearson_key_and_a_missing_metric_becomes_nan(self):
        result = ExperimentResult([make_run_result(metrics={"Pearson": 0.9, "RMSE": 1.0})])
        missing = ExperimentResult([make_run_result(metrics={"MSE": 0.5})])

        assert build_leaderboard_df(result)["PCC"].tolist() == [pytest.approx(0.9)]
        assert build_leaderboard_df(missing)["PCC"].isna().all()


class TestNormalized:
    """After ``normalize()`` the runs hold residuals; RMSE and raw Pearson come from ``raw_metrics``."""

    @pytest.fixture
    def df(self, experiment):
        return build_leaderboard_df(experiment.normalize(REFERENCE_MODEL)).set_index("algorithm")

    def test_pcc_is_normalized_while_rmse_and_raw_pearson_are_unnormalized(self, df, experiment):
        normalized_runs = _runs_of(experiment.normalize(REFERENCE_MODEL), "ElasticNet")
        runs = _runs_of(experiment, "ElasticNet")

        assert df.loc["ElasticNet", "PCC"] == pytest.approx(np.mean([r.metrics["Pearson"] for r in normalized_runs]))
        assert df.loc["ElasticNet", "RMSE"] == pytest.approx(np.mean([r.metrics["RMSE"] for r in runs]))
        assert df.loc["ElasticNet", "Pearson"] == pytest.approx(np.mean([r.metrics["Pearson"] for r in runs]))
        assert df["PCC"].notna().all()

    def test_per_drug_pearson_is_of_the_unnormalized_predictions(self, df, experiment):
        for name in ("RandomForest", REFERENCE_MODEL):
            expected = np.mean([_per_drug_pearson(r) for r in _runs_of(experiment, name)])
            assert df.loc[name, "Pearson_per_drug"] == pytest.approx(expected)

    def test_the_reference_model_is_a_baseline_with_zero_normalized_pcc_and_its_own_rmse(self, df, experiment):
        reference = df.loc[REFERENCE_MODEL]
        runs = _runs_of(experiment, REFERENCE_MODEL)

        assert bool(reference["is_baseline"]) is True
        assert reference["PCC"] == reference["PCC_std"] == 0.0
        assert reference["RMSE"] == pytest.approx(np.mean([r.metrics["RMSE"] for r in runs]))

    def test_runs_without_raw_metrics_leave_raw_pearson_and_per_drug_empty(self, experiment):
        normalized = experiment.normalize(REFERENCE_MODEL)
        for model in normalized.models:
            model.runs = [dataclasses.replace(r, raw_metrics={}, raw_per_drug_pearson=None) for r in model.runs]
        normalized.reference_runs = ()

        df = build_leaderboard_df(normalized)

        assert df["Pearson"].isna().all()
        assert df["Pearson_per_drug"].isna().all()
        assert df["RMSE"].notna().all()
