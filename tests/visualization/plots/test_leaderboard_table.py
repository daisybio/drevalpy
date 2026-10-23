"""Tests for :mod:`drevalpy.visualization.plots._leaderboard_table`.

The table is the one 1.6.1 ``load_results`` builds: normalized Pearson, RMSE, raw Pearson and
per-drug Pearson with their standard deviations over folds. ``tools/v1_parity/models/leaderboard.py``
compares it with 1.6.1 on identical predictions; these tests pin the same numbers by hand.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from drevalpy.evaluation import score_predictions
from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.types.results.run import RunResult
from drevalpy.visualization.plots._leaderboard_table import build_leaderboard_df
from tests.synthetic import REFERENCE_MODEL, make_experiment_result, make_run_result

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
    """A fold of 4 drugs x 6 samples whose per-drug and pooled correlations differ."""
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


class TestUnnormalized:
    def test_columns_are_the_aggregated_schema(self, experiment):
        assert list(build_leaderboard_df(experiment).columns) == COLUMNS

    def test_has_one_row_per_model(self, experiment):
        assert len(build_leaderboard_df(experiment)) == len(experiment.models)

    def test_rows_are_ordered_by_descending_pcc(self, experiment):
        assert build_leaderboard_df(experiment)["PCC"].is_monotonic_decreasing

    def test_naive_models_are_flagged_as_baselines(self, experiment):
        df = build_leaderboard_df(experiment).set_index("algorithm")

        assert bool(df.loc[REFERENCE_MODEL, "is_baseline"]) is True
        assert bool(df.loc["ElasticNet", "is_baseline"]) is False

    def test_raw_pearson_repeats_the_pcc_column(self, experiment):
        df = build_leaderboard_df(experiment)

        assert df["Pearson"].tolist() == df["PCC"].tolist()

    def test_the_means_are_taken_over_folds(self, experiment):
        row = build_leaderboard_df(experiment).set_index("algorithm").loc["ElasticNet"]
        runs = next(m for m in experiment.models if m.model_name == "ElasticNet").runs

        assert row["PCC"] == pytest.approx(np.mean([r.metrics["Pearson"] for r in runs]))
        assert row["RMSE"] == pytest.approx(np.mean([r.metrics["RMSE"] for r in runs]))

    def test_the_standard_deviation_is_the_sample_one(self, experiment):
        row = build_leaderboard_df(experiment).set_index("algorithm").loc["ElasticNet"]
        runs = next(m for m in experiment.models if m.model_name == "ElasticNet").runs

        assert row["PCC_std"] == pytest.approx(np.std([r.metrics["Pearson"] for r in runs], ddof=1))

    def test_per_drug_pearson_averages_the_drugs_then_the_folds(self, experiment):
        row = build_leaderboard_df(experiment).set_index("algorithm").loc["RandomForest"]
        runs = next(m for m in experiment.models if m.model_name == "RandomForest").runs
        per_fold = [_per_drug_pearson(r) for r in runs]

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

    def test_randomized_runs_are_excluded(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="ElasticNet", fold_index=0),
                make_run_result(model_name="ElasticNet", fold_index=1),
                make_run_result(model_name="RandomForest", randomization=("gene_expression", "permutation")),
            ]
        )

        assert build_leaderboard_df(result)["algorithm"].tolist() == ["ElasticNet"]

    def test_returns_the_empty_schema_when_every_run_is_randomized(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])

        df = build_leaderboard_df(result)

        assert df.empty
        assert list(df.columns) == COLUMNS

    def test_reads_the_plain_pearson_key(self):
        result = ExperimentResult([make_run_result(metrics={"Pearson": 0.9, "RMSE": 1.0})])

        assert build_leaderboard_df(result)["PCC"].tolist() == [pytest.approx(0.9)]

    def test_a_metric_no_run_reports_becomes_nan_rather_than_raising(self):
        result = ExperimentResult([make_run_result(metrics={"MSE": 0.5})])

        assert build_leaderboard_df(result)["PCC"].isna().all()


class TestNormalized:
    """After ``normalize()`` the runs hold residuals; RMSE and raw Pearson come from ``raw_metrics``."""

    @pytest.fixture
    def df(self, experiment):
        return build_leaderboard_df(experiment.normalize(REFERENCE_MODEL)).set_index("algorithm")

    def test_pcc_is_the_normalized_pearson(self, df, experiment):
        normalized = experiment.normalize(REFERENCE_MODEL)
        runs = next(m for m in normalized.models if m.model_name == "ElasticNet").runs

        assert df.loc["ElasticNet", "PCC"] == pytest.approx(np.mean([r.metrics["Pearson"] for r in runs]))

    def test_rmse_and_raw_pearson_are_the_unnormalized_ones(self, df, experiment):
        runs = next(m for m in experiment.models if m.model_name == "ElasticNet").runs

        assert df.loc["ElasticNet", "RMSE"] == pytest.approx(np.mean([r.metrics["RMSE"] for r in runs]))
        assert df.loc["ElasticNet", "Pearson"] == pytest.approx(np.mean([r.metrics["Pearson"] for r in runs]))

    def test_per_drug_pearson_is_of_the_unnormalized_predictions(self, df, experiment):
        runs = next(m for m in experiment.models if m.model_name == "RandomForest").runs

        expected = np.mean([_per_drug_pearson(r) for r in runs])
        assert df.loc["RandomForest", "Pearson_per_drug"] == pytest.approx(expected)

    def test_the_reference_model_is_listed_as_a_baseline_with_zero_normalized_pcc(self, df):
        reference = df.loc[REFERENCE_MODEL]

        assert bool(reference["is_baseline"]) is True
        assert reference["PCC"] == 0.0
        assert reference["PCC_std"] == 0.0

    def test_the_reference_model_keeps_its_unnormalized_metrics(self, df, experiment):
        runs = next(m for m in experiment.models if m.model_name == REFERENCE_MODEL).runs

        assert df.loc[REFERENCE_MODEL, "RMSE"] == pytest.approx(np.mean([r.metrics["RMSE"] for r in runs]))
        assert df.loc[REFERENCE_MODEL, "Pearson_per_drug"] == pytest.approx(
            np.mean([_per_drug_pearson(r) for r in runs])
        )

    def test_a_normalized_experiment_yields_finite_pcc_values(self):
        normalized = make_experiment_result(n_models=4, n_folds=3).normalize(REFERENCE_MODEL)

        assert build_leaderboard_df(normalized)["PCC"].notna().all()

    def test_runs_without_raw_metrics_leave_raw_pearson_and_per_drug_empty(self, experiment):
        normalized = experiment.normalize(REFERENCE_MODEL)
        for model in normalized.models:
            model.runs = [dataclasses.replace(r, raw_metrics={}, raw_per_drug_pearson=None) for r in model.runs]
        normalized.reference_runs = ()

        df = build_leaderboard_df(normalized)

        assert df["Pearson"].isna().all()
        assert df["Pearson_per_drug"].isna().all()
        assert df["RMSE"].notna().all()
