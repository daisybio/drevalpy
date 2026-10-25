"""The summary tables of an experiment: pooled, grouped, and the group counts behind them."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.evaluation import evaluate
from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.types.results.run import RunResult
from tests.synthetic import make_experiment_result, make_run_result


def _drug_run(
    model_name: str, predictions: list[float], *, fold_index: int = 0, response: list[float] | None = None
) -> RunResult:
    """Two drugs of four pairs (response 0..3 for A, 10..13 for B by default); each sample appears once per drug."""
    run = make_run_result(model_name=model_name, fold_index=fold_index, n_pairs=8, n_samples=4, n_drugs=2)
    run.drug_ids = np.array(["A"] * 4 + ["B"] * 4)
    run.sample_ids = np.array([f"S{i % 4}" for i in range(8)])
    run.ground_truth = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0] if response is None else response)
    run.predictions = np.array(predictions)
    run.metrics = evaluate(run.predictions, run.ground_truth, ["MSE", "Pearson"])
    return run


class TestGroupedSummaryTable:
    def test_the_drug_mean_baseline_is_high_pooled_and_zero_per_drug(self) -> None:
        experiment = ExperimentResult(
            [_drug_run("DrugMean", [1.5] * 4 + [11.5] * 4), _drug_run("Varying", [0.0, 2.0, 1.0, 3.0, 10, 11, 12, 13])]
        )
        pooled = experiment.summary_table().loc["DrugMean", "Pearson"]
        per_drug = experiment.summary_table(group_by="drug").loc["DrugMean", "Pearson"]

        assert pooled > 0.95
        assert per_drug == 0.0
        counts = experiment.summary_group_counts(group_by="drug")
        assert counts.loc["DrugMean", "Pearson"] == counts.loc["Varying", "Pearson"] == 2

    def test_each_metric_is_averaged_over_the_drugs(self) -> None:
        experiment = ExperimentResult([_drug_run("M", [0.0, 2.0, 1.0, 3.0, 10.0, 11.0, 12.0, 13.0])])

        table = experiment.summary_table(group_by="drug").loc["M"]

        assert table["Pearson"] == pytest.approx((0.8 + 1.0) / 2)
        assert table["MSE"] == pytest.approx((0.5 + 0.0) / 2)

    def test_a_group_with_a_constant_prediction_scores_zero_and_is_counted(self) -> None:
        experiment = ExperimentResult([_drug_run("M", [1.0] * 4 + [10.0, 11.0, 12.0, 13.0])])

        assert experiment.summary_table(group_by="drug").loc["M", "Pearson"] == pytest.approx(0.5)
        assert experiment.summary_group_counts(group_by="drug").loc["M"].to_dict() == {"MSE": 2.0, "Pearson": 2.0}

    def test_a_group_with_a_constant_response_is_skipped_not_scored_zero(self) -> None:
        response = [5.0] * 4 + [10.0, 11.0, 12.0, 13.0]
        experiment = ExperimentResult([_drug_run("M", [0.0, 2.0, 1.0, 3.0, 10.0, 11.0, 12.0, 13.0], response=response)])

        assert experiment.summary_table(group_by="drug").loc["M", "Pearson"] == pytest.approx(1.0)
        assert experiment.summary_group_counts(group_by="drug").loc["M"].to_dict() == {"MSE": 2.0, "Pearson": 1.0}

    def test_folds_are_averaged_after_the_groups(self) -> None:
        runs = [
            _drug_run("M", [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0], fold_index=0),
            _drug_run("M", [0.0, 2.0, 1.0, 3.0, 10.0, 12.0, 11.0, 13.0], fold_index=1),
        ]

        assert ExperimentResult(runs).summary_table(group_by="drug").loc["M", "Pearson"] == pytest.approx(
            (1.0 + 0.8) / 2
        )

    def test_a_fold_where_nothing_is_defined_is_skipped(self) -> None:
        runs = [
            _drug_run("M", [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0], fold_index=0, response=[2.0] * 4 + [7.0] * 4),
            _drug_run("M", [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0], fold_index=1),
        ]

        assert ExperimentResult(runs).summary_table(group_by="drug").loc["M", "Pearson"] == pytest.approx(1.0)

    def test_a_fold_with_a_constant_prediction_is_averaged_in_as_zero(self) -> None:
        runs = [
            _drug_run("M", [1.0] * 4 + [11.0] * 4, fold_index=0),
            _drug_run("M", [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0], fold_index=1),
        ]

        assert ExperimentResult(runs).summary_table(group_by="drug").loc["M", "Pearson"] == pytest.approx(0.5)

    def test_sample_grouping_uses_the_sample_ids(self) -> None:
        run = _drug_run("M", [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])

        table = ExperimentResult([run]).summary_table(group_by="sample").loc["M"]

        assert table["MSE"] == pytest.approx(0.0)
        assert table["Pearson"] == pytest.approx(1.0)

    def test_study_grouping_gives_one_group_per_sample_id_prefix(self) -> None:
        runs = []
        for fold in range(2):
            run = _drug_run("M", [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0], fold_index=fold)
            run.sample_ids = np.array([f"{'A' if i % 4 < 2 else 'B'}::S{i % 4}" for i in range(8)])
            runs.append(run)
        experiment = ExperimentResult(runs)

        table = experiment.summary_table(group_by="study").loc["M"]

        assert table["MSE"] == pytest.approx(0.0)
        assert table["Pearson"] == pytest.approx(1.0)
        assert experiment.summary_group_counts(group_by="study").loc["M"].to_dict() == {"MSE": 2.0, "Pearson": 2.0}

    def test_study_grouping_without_prefixes_is_undefined(self) -> None:
        run = _drug_run("M", [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
        experiment = ExperimentResult([run])
        table = experiment.summary_table(group_by="study").loc["M"]
        assert table.isna().all()
        assert experiment.summary_group_counts(group_by="study").loc["M"].to_dict() == {"MSE": 0.0, "Pearson": 0.0}

    def test_study_grouping_scores_each_study_separately(self) -> None:
        run = _drug_run("M", [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
        run.sample_ids = np.array(["A::S0"] * 4 + ["B::S0"] * 4)
        run.predictions = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 15.0])
        run.metrics = evaluate(run.predictions, run.ground_truth, ["MSE"])

        table = ExperimentResult([run]).summary_table(group_by="study").loc["M"]

        assert table["MSE"] == pytest.approx((0.0 + 1.0) / 2)

    def test_an_unknown_grouping_is_rejected(self) -> None:
        experiment = make_experiment_result(n_models=1, n_folds=1)

        with pytest.raises(ValueError, match="Unknown grouping 'tissue'"):
            experiment.summary_table(group_by="tissue")  # type: ignore[arg-type]

    def test_normalized_results_support_grouping(self) -> None:
        experiment = make_experiment_result(n_models=3, n_folds=2, n_pairs=40)
        pooled = experiment.normalize().summary_table()

        per_drug = experiment.normalize().summary_table(group_by="drug")

        assert list(per_drug.index) == list(pooled.index)
        assert np.isfinite(per_drug.to_numpy()).all()

    def test_randomized_runs_do_not_enter_the_grouped_table(self) -> None:
        ordinary = _drug_run("M", [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
        randomized = _drug_run("M", [3.0, 2.0, 1.0, 0.0, 13.0, 12.0, 11.0, 10.0], fold_index=1)
        randomized.randomization = ("gene_expression", "permutation")

        assert ExperimentResult([ordinary, randomized]).summary_table(group_by="drug").loc["M", "Pearson"] == 1.0


class TestPooledSummaryTable:
    def test_is_a_method_returning_the_mean_of_each_metric(self) -> None:
        runs = [
            make_run_result(model_name="A", fold_index=0, metrics={"MSE": 0.2}),
            make_run_result(model_name="A", fold_index=1, metrics={"MSE": 0.4}),
        ]

        table = ExperimentResult(runs).summary_table()

        assert list(table.index) == ["A"]
        assert list(table.columns) == ["MSE"]
        assert table.loc["A", "MSE"] == pytest.approx(0.3)
