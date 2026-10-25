"""Tests for ``drevalpy.evaluation._scoring``: pooled scores, per-group scores and their mean."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from drevalpy.evaluation import Metric, PerGroupScores, evaluate, evaluate_per_group
from drevalpy.evaluation._pairs import CONSTANT_ATOL
from drevalpy.evaluation._scoring import mean_of_defined, run_metrics, run_per_group_score
from tests.synthetic import make_run_result


def test_evaluate() -> None:
    predictions = np.array([1, 2, 3, 4, 5])
    response = np.array([1.1, 2.2, 3.3, 4.4, 5.5])

    results = evaluate(predictions, response, ["MSE", "RMSE", "MAE", "R^2"])

    assert np.isclose(results["MSE"], mean_squared_error(predictions, response))
    assert np.isclose(results["RMSE"], np.sqrt(mean_squared_error(predictions, response)))
    assert np.isclose(results["MAE"], mean_absolute_error(predictions, response))
    assert np.isclose(results["R^2"], r2_score(y_pred=predictions, y_true=response))


def test_evaluate_without_metrics_scores_every_metric_in_report_order() -> None:
    scores = evaluate(np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 4.0]))

    assert list(scores) == list(Metric)


def test_evaluate_accepts_a_single_name_a_single_member_and_an_iterable() -> None:
    predictions, response = np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 4.0])
    expected = evaluate(predictions, response)

    assert evaluate(predictions, response, "Pearson") == {"Pearson": expected["Pearson"]}
    assert evaluate(predictions, response, Metric.PEARSON) == {"Pearson": expected["Pearson"]}
    assert evaluate(predictions, response, (metric for metric in (Metric.MAE, "RMSE"))) == {
        "MAE": expected["MAE"],
        "RMSE": expected["RMSE"],
    }
    assert list(evaluate(predictions, response, [Metric.KENDALL, Metric.MSE])) == ["Kendall", "MSE"]


def test_evaluate_with_all_nan_predictions() -> None:
    assert np.isnan(evaluate(np.array([np.nan, np.nan]), np.array([1.0, 2.0]), ["RMSE"])["RMSE"])


@pytest.mark.parametrize("metrics", ["nope", ["nope"], ["MSE", "nope"]])
def test_an_unknown_metric_raises_value_error(metrics) -> None:
    with pytest.raises(ValueError, match="Unknown metric 'nope'. Available: MSE, RMSE"):
        evaluate(np.array([1.0, 2.0]), np.array([1.0, 2.0]), metrics)


@pytest.mark.parametrize("metric, expected", [("MSE", 6.25), ("RMSE", 2.5), ("MAE", 2.5)])
def test_error_metrics_score_one_jointly_valid_pair(metric, expected):
    assert evaluate(np.array([1.5, np.nan, 2.0]), np.array([4.0, 1.0, np.nan]), metric)[metric] == expected


@pytest.mark.parametrize("metric", ["R^2", "Pearson", "Spearman", "Kendall"])
def test_statistical_metrics_require_two_jointly_valid_pairs(metric):
    assert np.isnan(evaluate(np.array([1.5]), np.array([4.0]), metric)[metric])


@pytest.mark.parametrize("metric", ["R^2", "Pearson", "Spearman", "Kendall"])
def test_a_constant_response_is_undefined_even_for_a_constant_prediction(metric):
    for predictions in (np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0])):
        assert np.isnan(evaluate(predictions, np.array([5.0, 5.0, 5.0]), metric)[metric])


def test_a_constant_response_leaves_the_error_metrics_defined() -> None:
    scores = evaluate(np.array([4.0, 5.0, 8.0]), np.array([5.0, 5.0, 5.0]), ["MSE", "MAE"])

    assert scores == {"MSE": pytest.approx(10 / 3), "MAE": pytest.approx(4 / 3)}


@pytest.mark.parametrize("metric", ["Pearson", "Spearman", "Kendall"])
def test_a_constant_prediction_scores_zero_for_a_correlation(metric):
    assert evaluate(np.full(3, 2.0), np.array([1.0, 2.0, 4.0]), metric)[metric] == 0.0


def test_a_constant_prediction_keeps_the_r2_formula() -> None:
    response = np.array([1.0, 2.0, 4.0])
    predictions = np.full(3, 2.0)

    assert evaluate(predictions, response, "R^2")["R^2"] == pytest.approx(r2_score(response, predictions))


@pytest.mark.parametrize("predictions, targets", [(np.ones(2), np.ones(3)), (np.ones((2, 1)), np.ones(2))])
def test_scoring_rejects_alignment_errors(predictions, targets):
    with pytest.raises(ValueError, match="one-dimensional and equally long"):
        evaluate(predictions, targets)
    with pytest.raises(ValueError, match="one-dimensional and equally long"):
        run_metrics(predictions, targets)


class TestRunMetrics:
    def test_a_run_without_a_usable_pair_records_no_metrics(self):
        assert run_metrics(np.array([np.nan, 2.0]), np.array([1.0, np.nan])) == {}

    def test_a_run_with_usable_pairs_records_every_metric_as_evaluate_does(self):
        predictions, response = np.array([1.0, 2.0, 4.0, np.nan]), np.array([1.0, 3.0, 4.5, 2.0])

        scores = run_metrics(predictions, response)

        assert list(scores) == list(Metric)
        assert scores == evaluate(predictions, response)

    def test_a_single_pair_records_nan_for_the_metrics_it_cannot_support(self):
        scores = run_metrics(np.array([1.0]), np.array([3.0]))

        assert scores["MSE"] == 4.0
        assert np.isnan(scores["Pearson"])


CORRELATIONS = ["Pearson", "Spearman", "Kendall"]


@pytest.fixture(scope="module")
def pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels, predictions and response of six groups, with ties in both variables."""
    rng = np.random.default_rng(3)
    labels = np.repeat([f"D{i}" for i in range(6)], 15)
    response = np.round(rng.normal(size=labels.size), 1)
    predictions = np.round(response + rng.normal(size=labels.size), 1)
    return labels, predictions, response


class TestEvaluatePerGroup:
    def test_pairs_without_a_group_or_with_non_finite_values_are_dropped(self):
        labels = np.array(["A", "A", np.nan, "B", "B", np.nan, "B"], dtype=object)
        predictions = np.array([1.0, 2.0, 100.0, 1.0, 3.0, -100.0, np.nan])
        response = np.array([1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 4.0])

        scores = evaluate_per_group(predictions, response, "MSE", labels=labels)

        assert scores.groups.tolist() == ["A", "B"]
        assert scores.n_pairs.tolist() == [2, 2]
        np.testing.assert_allclose(scores.values["MSE"], [0.0, 0.5])

    def test_labels_that_are_all_undefined_leave_no_groups(self):
        labels = np.array([np.nan, np.nan], dtype=object)

        scores = evaluate_per_group(np.array([1.0, 2.0]), np.array([1.0, 3.0]), "MSE", labels=labels)

        assert scores.groups.size == scores.n_pairs.size == scores.values["MSE"].size == 0
        assert np.isnan(scores.mean()["MSE"])
        assert scores.n_defined() == {"MSE": 0}

    def test_every_metric_matches_scipy_and_sklearn_per_group(self, pairs):
        labels, predictions, response = pairs
        references = {
            "MSE": lambda p, r: mean_squared_error(r, p),
            "RMSE": lambda p, r: np.sqrt(mean_squared_error(r, p)),
            "MAE": lambda p, r: mean_absolute_error(r, p),
            "R^2": lambda p, r: r2_score(r, p),
            "Pearson": lambda p, r: pearsonr(p, r)[0],
            "Spearman": lambda p, r: spearmanr(p, r)[0],
            "Kendall": lambda p, r: kendalltau(p, r)[0],
        }

        scores = evaluate_per_group(predictions, response, labels=labels)

        assert set(scores.values) == set(Metric) == set(references)
        assert list(scores.groups) == [f"D{i}" for i in range(6)]
        assert scores.n_pairs.tolist() == [15] * 6
        for metric, reference in references.items():
            for index, group in enumerate(scores.groups):
                mask = labels == group
                expected = reference(predictions[mask], response[mask])
                assert scores.values[metric][index] == pytest.approx(expected, abs=1e-9)
                pooled = evaluate(predictions[mask], response[mask], metric)[metric]
                assert scores.values[metric][index] == pytest.approx(pooled)

    def test_the_requested_metrics_are_the_only_ones_returned(self, pairs):
        labels, predictions, response = pairs

        scores = evaluate_per_group(predictions, response, ["RMSE", Metric.KENDALL], labels=labels)

        assert list(scores.values) == ["RMSE", "Kendall"]

    def test_an_unknown_metric_and_mismatched_lengths_are_rejected(self, pairs):
        labels, predictions, response = pairs
        with pytest.raises(ValueError, match="Unknown metric 'Nope'"):
            evaluate_per_group(predictions, response, ["Nope"], labels=labels)
        with pytest.raises(ValueError, match="equally long"):
            evaluate_per_group(np.array([1.0]), np.array([1.0, 2.0]), labels=np.array(["a", "b"]))
        with pytest.raises(ValueError, match="equally long"):
            evaluate_per_group(np.ones(2), np.ones(2), labels=np.array([["a", "b"]]))


class TestFixedGroupAxis:
    labels = np.array(["B", "B", "D", "D", "D", "Z"])
    predictions = np.array([1.0, 2.0, 1.0, 3.0, 2.0, 5.0])
    response = np.array([1.0, 3.0, 2.0, 1.0, 4.0, 5.0])
    axis = np.array(["A", "B", "C", "D"])

    def test_groups_without_pairs_are_kept_and_pairs_off_the_axis_are_dropped(self):
        scores = evaluate_per_group(self.predictions, self.response, labels=self.labels, groups=self.axis)

        assert scores.groups.tolist() == ["A", "B", "C", "D"]
        assert scores.n_pairs.tolist() == [0, 2, 0, 3]
        for metric in Metric:
            values = scores.values[metric]
            assert np.isnan(values[[0, 2]]).all()
            assert np.isfinite(values[[1, 3]]).all()

    def test_scores_on_the_axis_equal_those_of_the_free_axis(self):
        free = evaluate_per_group(self.predictions, self.response, labels=self.labels)
        fixed = evaluate_per_group(self.predictions, self.response, labels=self.labels, groups=self.axis)

        assert free.groups.tolist() == ["B", "D", "Z"]
        for metric in Metric:
            np.testing.assert_array_equal(fixed.values[metric][[1, 3]], free.values[metric][[0, 1]])

    def test_two_models_share_the_axis_and_the_mean_skips_the_empty_groups(self):
        first = evaluate_per_group(self.predictions, self.response, "MSE", labels=self.labels, groups=self.axis)
        second = evaluate_per_group(self.response, self.response, "MSE", labels=self.labels, groups=self.axis)

        assert first.groups.tolist() == second.groups.tolist() == self.axis.tolist()
        assert first.n_defined() == {"MSE": 2}
        assert second.mean() == {"MSE": 0.0}

    def test_an_empty_axis_has_no_groups(self):
        scores = evaluate_per_group(self.predictions, self.response, "MSE", labels=self.labels, groups=np.array([]))

        assert scores.groups.size == scores.n_pairs.size == 0


class TestPerGroupScoresResult:
    scores = PerGroupScores(
        groups=np.array(["A", "B", "C"]),
        n_pairs=np.array([3, 1, 0]),
        values={"MSE": np.array([0.5, 1.5, np.nan]), "Pearson": np.array([np.nan, np.nan, np.nan])},
    )

    def test_mean_skips_undefined_groups_and_is_nan_without_a_defined_one(self):
        mean = self.scores.mean()

        assert mean["MSE"] == pytest.approx(1.0)
        assert np.isnan(mean["Pearson"])
        assert list(mean) == ["MSE", "Pearson"]

    def test_n_defined_counts_the_groups_behind_each_mean(self):
        assert self.scores.n_defined() == {"MSE": 2, "Pearson": 0}

    def test_a_zero_score_is_defined(self):
        scores = PerGroupScores(np.array(["A", "B"]), np.array([2, 2]), {"Pearson": np.array([0.0, 0.5])})

        assert scores.mean()["Pearson"] == pytest.approx(0.25)
        assert scores.n_defined() == {"Pearson": 2}


class TestUndefinedGroups:
    labels = np.repeat(["const_pred", "const_true", "varying", "single"], [4, 4, 4, 1])
    predictions = np.array([1.0] * 4 + [1.0, 2.0, 3.0, 4.0] + [1.0, 2.0, 4.0, 3.0] + [7.0])
    response = np.array([1.0, 2.0, 3.0, 4.0] + [2.0] * 4 + [1.0, 2.0, 3.0, 4.0] + [7.0])

    @pytest.fixture(scope="class")
    def scores(self) -> dict[str, np.ndarray]:
        result = evaluate_per_group(self.predictions, self.response, labels=self.labels)
        assert list(result.groups) == ["const_pred", "const_true", "single", "varying"]
        return result.values

    @pytest.mark.parametrize(
        ("group", "metrics"),
        [
            pytest.param(1, [*CORRELATIONS, "R^2"], id="constant-response"),
            pytest.param(2, [*CORRELATIONS, "R^2"], id="single-pair"),
        ],
    )
    def test_the_data_makes_the_group_undefined(self, scores, group, metrics):
        assert all(np.isnan(scores[metric][group]) for metric in metrics)

    @pytest.mark.parametrize("metric", CORRELATIONS)
    def test_a_constant_prediction_scores_zero_for_a_correlation(self, scores, metric):
        assert scores[metric][0] == 0.0

    def test_a_constant_prediction_keeps_the_r2_formula(self, scores):
        assert scores["R^2"][0] == pytest.approx(1 - 14 / 5)

    def test_error_metrics_and_a_varying_group_are_scored(self, scores):
        assert all(np.isfinite(scores[metric]).all() for metric in ("MSE", "RMSE", "MAE"))
        assert scores["Pearson"][3] == pytest.approx(0.8)

    @pytest.mark.parametrize(
        "predictions",
        [
            pytest.param(5.0 + np.array([0.0, CONSTANT_ATOL / 10, 0.0, CONSTANT_ATOL / 10]), id="nearly-constant"),
            pytest.param(np.full(4, 1234.5678901), id="large-constant"),
        ],
    )
    def test_a_constant_prediction_is_not_mistaken_for_variation(self, predictions):
        scores = evaluate_per_group(predictions, np.arange(4.0), CORRELATIONS, labels=np.zeros(4, dtype=int))

        assert [scores.values[metric][0] for metric in CORRELATIONS] == [0.0, 0.0, 0.0]


class TestAModelThatIsConstantInOneGroup:
    labels = np.array(["A"] * 4 + ["B"] * 4)
    response = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 6.0, 8.0])
    varying = np.array([1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    constant_in_a = np.array([2.0] * 4 + [5.0, 6.0, 7.0, 8.0])

    @pytest.mark.parametrize("metric", CORRELATIONS)
    def test_the_constant_group_scores_zero_and_the_other_group_is_unchanged(self, metric):
        varying = evaluate_per_group(self.varying, self.response, metric, labels=self.labels).values
        constant = evaluate_per_group(self.constant_in_a, self.response, metric, labels=self.labels).values

        assert constant[metric][0] == 0.0
        assert varying[metric][0] > 0.5
        assert constant[metric][1] == varying[metric][1]

    @pytest.mark.parametrize("metric", [*CORRELATIONS, "R^2", "MSE"])
    def test_both_models_are_averaged_over_the_same_two_groups(self, metric):
        counts = [
            evaluate_per_group(predictions, self.response, metric, labels=self.labels).n_defined()[metric]
            for predictions in (self.varying, self.constant_in_a)
        ]

        assert counts == [2, 2]

    def test_the_mean_includes_the_zero(self):
        scores = evaluate_per_group(self.constant_in_a, self.response, "Pearson", labels=self.labels)

        assert scores.mean()["Pearson"] == pytest.approx(scores.values["Pearson"][1] / 2)


class TestMeanOfDefined:
    def test_undefined_groups_are_skipped_not_counted_as_zero(self):
        assert mean_of_defined(np.array([0.5, np.nan, 0.3])) == pytest.approx(0.4)

    def test_a_zero_score_is_counted(self):
        assert mean_of_defined(np.array([0.0, 0.5])) == pytest.approx(0.25)

    def test_non_finite_values_are_skipped(self):
        assert mean_of_defined([1.0, np.inf, 3.0]) == pytest.approx(2.0)

    @pytest.mark.parametrize("values", [np.array([np.nan, np.nan]), np.array([])])
    def test_no_defined_group_gives_nan(self, values):
        assert np.isnan(mean_of_defined(values))


class TestRunPerGroupScore:
    def test_averages_the_defined_drugs_of_a_run(self):
        run = make_run_result(n_pairs=20, n_drugs=4)
        per_drug = [
            np.corrcoef(run.predictions[run.drug_ids == d], run.ground_truth[run.drug_ids == d])[0, 1]
            for d in np.unique(run.drug_ids)
        ]

        assert run_per_group_score(run, "drug", "Pearson") == pytest.approx(np.mean(per_drug))

    def test_a_drug_with_a_constant_prediction_is_counted_as_zero(self):
        run = make_run_result(n_pairs=20, n_drugs=2)
        constant = run.drug_ids == "D_0"
        varying_score = run_per_group_score(run, "drug", "Pearson")
        only_d1 = np.corrcoef(run.predictions[~constant], run.ground_truth[~constant])[0, 1]
        run.predictions = np.where(constant, 0.5, run.predictions)

        assert run_per_group_score(run, "drug", "Pearson") == pytest.approx(only_d1 / 2)
        assert run_per_group_score(run, "drug", "Pearson") != varying_score

    def test_a_run_without_a_defined_group_is_nan_and_an_unknown_grouping_is_rejected(self):
        assert np.isnan(run_per_group_score(make_run_result(n_pairs=1, n_drugs=1, n_samples=1), "drug", "Pearson"))
        with pytest.raises(ValueError, match="Unknown grouping"):
            run_per_group_score(make_run_result(), "tissue", "Pearson")  # type: ignore[arg-type]
