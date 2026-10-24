"""Tests for metric evaluation: ``drevalpy.evaluation``, grouped scoring and the grouped Pearson kernel."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from drevalpy._grouped_evaluation import (
    GROUPINGS,
    _group_score,
    defined_labels,
    group_labels,
    grouped_scores,
    mean_over_groups,
    run_group_score,
)
from drevalpy._grouped_pearson import PearsonSums
from drevalpy.evaluation import (
    AVAILABLE_METRICS,
    CONSTANT_ATOL,
    evaluate,
    is_constant,
    kendall,
    pearson,
    score_predictions,
    spearman,
)
from tests.synthetic import make_run_result

CORRELATIONS = [pearson, spearman, kendall]


N = 50_000


RESPONSE = np.arange(float(N))


def test_evaluate() -> None:
    predictions = np.array([1, 2, 3, 4, 5])
    response = np.array([1.1, 2.2, 3.3, 4.4, 5.5])

    results = evaluate(predictions, response, metric=["MSE", "RMSE", "MAE", "R^2"])

    assert np.isclose(results["MSE"], mean_squared_error(predictions, response))
    assert np.isclose(results["RMSE"], np.sqrt(mean_squared_error(predictions, response)))
    assert np.isclose(results["MAE"], mean_absolute_error(predictions, response))
    assert np.isclose(results["R^2"], r2_score(y_pred=predictions, y_true=response))


@pytest.mark.parametrize("correlation", CORRELATIONS)
def test_correlation_extremes(correlation) -> None:
    uncorrelated = np.random.default_rng(0).permutation(RESPONSE)

    assert np.isclose(correlation(RESPONSE, RESPONSE), 1.0, atol=1e-3)
    assert np.isclose(correlation(RESPONSE[::-1], RESPONSE), -1.0, atol=1e-3)
    assert np.isclose(correlation(uncorrelated, RESPONSE), 0.0, atol=2e-2)
    assert np.isclose(correlation(np.ones(N), RESPONSE), 0.0, atol=1e-3)


@pytest.mark.parametrize("correlation", CORRELATIONS)
def test_correlations_raise_value_error_on_length_mismatch(correlation) -> None:
    with pytest.raises(ValueError, match="same length"):
        correlation(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_score_predictions_all_nan_predictions() -> None:
    assert np.isnan(score_predictions(np.array([np.nan, np.nan]), np.array([1.0, 2.0]), ["RMSE"])["RMSE"])


@pytest.mark.parametrize("function", [score_predictions, evaluate])
def test_an_unknown_metric_raises_value_error(function) -> None:
    metric = ["nope"] if function is score_predictions else "nope"
    with pytest.raises(ValueError, match="invalid metric"):
        function(np.array([1.0, 2.0]), np.array([1.0, 2.0]), metric)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="nothing"),
        pytest.param({"predictions": np.array([1.0, 2.0])}, id="predictions-without-response"),
        pytest.param({"predictions_or_dataset": np.array([1.0, 2.0])}, id="positional-predictions-without-response"),
        pytest.param({"response": np.array([1.0, 2.0])}, id="response-without-predictions"),
    ],
)
def test_evaluate_raises_value_error_when_predictions_or_response_are_missing(kwargs) -> None:
    with pytest.raises(ValueError, match="needs predictions and response"):
        evaluate(**kwargs)


def test_evaluate_reads_predictions_and_response_from_a_dataset_object() -> None:
    class _Carrier:
        predictions = np.array([1.0, 2.0, 3.0])
        response = np.array([1.5, 2.0, 2.5])

    assert evaluate(_Carrier(), metric="MAE")["MAE"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("metric, expected", [("MSE", 6.25), ("RMSE", 2.5), ("MAE", 2.5)])
def test_error_metrics_score_one_jointly_valid_pair(metric, expected):
    assert score_predictions(np.array([1.5, np.nan, 2.0]), np.array([4.0, 1.0, np.nan]), [metric])[metric] == expected


@pytest.mark.parametrize("metric", ["R^2", "Pearson", "Spearman", "Kendall"])
def test_statistical_metrics_require_two_jointly_valid_pairs(metric):
    assert np.isnan(score_predictions(np.array([1.5]), np.array([4.0]), [metric])[metric])


@pytest.mark.parametrize("predictions, targets", [(np.ones(2), np.ones(3)), (np.ones((2, 1)), np.ones(2))])
def test_scoring_rejects_alignment_errors(predictions, targets):
    with pytest.raises(ValueError, match="equal-shaped one-dimensional"):
        score_predictions(predictions, targets)


def test_empty_scoring_conventions_share_the_same_mask():
    predictions, targets = np.array([np.nan, 2.0]), np.array([1.0, np.nan])
    assert score_predictions(predictions, targets, empty_as_dict=True) == {}
    assert all(np.isnan(v) for v in score_predictions(predictions, targets).values())


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.full(5, 2.5), True),
        (np.array([0.0, 5e-7]), True),
        (np.array([0.0, 2e-6]), False),
        (np.array([1e5, 1e5 + 0.5]), True),
        (np.array([1e5, 1e5 + 5.0]), False),
        (np.array([-1e5, -1e5 + 0.5]), True),
        (np.array([1.0, 2.0, 3.0]), False),
        (np.array([7.0]), True),
    ],
)
def test_is_constant_edge_cases(values: np.ndarray, expected: bool) -> None:
    assert is_constant(values) is expected


def test_is_constant_does_not_depend_on_order() -> None:
    """Comparing every value with ``values[0]`` used to make the verdict depend on the first one."""
    rng = np.random.default_rng(0)
    for values in (np.array([0.0, 5e-7, 5e-7, 0.0]), 1e5 + np.array([0.0, 0.9, 0.3]), np.array([-1e5, 1e5, 0.0])):
        assert len({is_constant(rng.permutation(values)) for _ in range(30)}) == 1


@pytest.mark.parametrize("correlation", CORRELATIONS)
def test_correlation_verdicts_do_not_depend_on_order(correlation) -> None:
    rng = np.random.default_rng(1)
    near_constant = 1e5 + np.array([0.0, 0.9, 0.3, 0.6])
    varying = np.arange(4.0)
    for _ in range(20):
        order = rng.permutation(4)
        assert correlation(near_constant[order], varying[order]) == 0.0
        assert np.isnan(correlation(varying[order], near_constant[order]))


@pytest.fixture(scope="module")
def pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels, predictions and response of six groups, with ties in both variables."""
    rng = np.random.default_rng(3)
    labels = np.repeat([f"D{i}" for i in range(6)], 15)
    response = np.round(rng.normal(size=labels.size), 1)
    predictions = np.round(response + rng.normal(size=labels.size), 1)
    return labels, predictions, response


class TestGroupLabels:
    def test_drug_sample_and_study_groupings(self):
        run = make_run_result(n_pairs=4)
        run.sample_ids = np.array(["GDSC2::CL_1", "CTRPv2::CL_1", "GDSC2::a::b", "CTRPv2::CL_2"])

        assert GROUPINGS == ("drug", "sample", "study")
        np.testing.assert_array_equal(group_labels(run, "drug"), run.drug_ids)
        np.testing.assert_array_equal(group_labels(run, "sample"), run.sample_ids)
        assert group_labels(run, "study").tolist() == ["GDSC2", "CTRPv2", "GDSC2", "CTRPv2"]
        assert defined_labels(group_labels(run, "drug")).all()

    @pytest.mark.parametrize("sample_id", ["CL_1", "::CL_1"])
    def test_a_sample_id_without_a_study_prefix_has_no_study(self, sample_id):
        run = make_run_result(n_pairs=2)
        run.sample_ids = np.array(["GDSC2::CL_1", sample_id])

        labels = group_labels(run, "study")

        assert labels[0] == "GDSC2"
        assert labels[1] is np.nan
        assert defined_labels(labels).tolist() == [True, False]

    def test_an_unknown_grouping_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown grouping 'tissue'"):
            group_labels(make_run_result(), "tissue")


class TestGroupedScores:
    def test_pairs_without_a_group_or_with_non_finite_values_are_dropped(self):
        labels = np.array(["A", "A", np.nan, "B", "B", np.nan, "B"], dtype=object)
        predictions = np.array([1.0, 2.0, 100.0, 1.0, 3.0, -100.0, np.nan])
        response = np.array([1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 4.0])

        groups, scores = grouped_scores(labels, predictions, response, ["MSE"])

        assert groups.tolist() == ["A", "B"]
        np.testing.assert_allclose(scores["MSE"], [0.0, 0.5])

    def test_labels_that_are_all_undefined_leave_no_groups(self):
        labels = np.array([np.nan, np.nan], dtype=object)

        groups, scores = grouped_scores(labels, np.array([1.0, 2.0]), np.array([1.0, 3.0]), ["MSE"])

        assert groups.size == scores["MSE"].size == 0

    def test_every_metric_matches_the_registered_function_per_group(self, pairs):
        labels, predictions, response = pairs

        groups, scores = grouped_scores(labels, predictions, response)

        assert set(scores) == set(AVAILABLE_METRICS)
        assert list(groups) == [f"D{i}" for i in range(6)]
        for metric, function in AVAILABLE_METRICS.items():
            for index, group in enumerate(groups):
                mask = labels == group
                assert scores[metric][index] == pytest.approx(
                    _group_score(metric, predictions[mask], response[mask]), abs=1e-9
                )
                assert scores[metric][index] == pytest.approx(function(y_pred=predictions[mask], y_true=response[mask]))

    def test_an_unknown_metric_and_mismatched_lengths_are_rejected(self, pairs):
        with pytest.raises(ValueError, match="invalid metric Nope"):
            grouped_scores(*pairs, ["Nope"])
        with pytest.raises(ValueError, match="equally long"):
            grouped_scores(np.array(["a", "b"]), np.array([1.0]), np.array([1.0, 2.0]))


class TestUndefinedGroups:
    labels = np.repeat(["const_pred", "const_true", "varying", "single"], [4, 4, 4, 1])
    predictions = np.array([1.0] * 4 + [1.0, 2.0, 3.0, 4.0] + [1.0, 2.0, 4.0, 3.0] + [7.0])
    response = np.array([1.0, 2.0, 3.0, 4.0] + [2.0] * 4 + [1.0, 2.0, 3.0, 4.0] + [7.0])

    @pytest.fixture(scope="class")
    def scores(self) -> dict[str, np.ndarray]:
        groups, scores = grouped_scores(self.labels, self.predictions, self.response)
        assert list(groups) == ["const_pred", "const_true", "single", "varying"]
        return scores

    @pytest.mark.parametrize(
        ("group", "metrics"),
        [
            pytest.param(0, ["Pearson", "Spearman", "Kendall"], id="constant-prediction"),
            pytest.param(1, ["Pearson", "Spearman", "Kendall", "R^2"], id="constant-response"),
            pytest.param(2, ["Pearson", "Spearman", "Kendall", "R^2"], id="single-pair"),
        ],
    )
    def test_the_group_is_undefined(self, scores, group, metrics):
        assert all(np.isnan(scores[metric][group]) for metric in metrics)

    def test_error_metrics_and_a_varying_group_are_scored(self, scores):
        assert all(np.isfinite(scores[metric]).all() for metric in ("MSE", "RMSE", "MAE"))
        assert np.isfinite(scores["R^2"][0])
        assert scores["Pearson"][3] == pytest.approx(0.8)

    @pytest.mark.parametrize(
        "predictions",
        [
            pytest.param(5.0 + np.array([0.0, CONSTANT_ATOL / 10, 0.0, CONSTANT_ATOL / 10]), id="nearly-constant"),
            pytest.param(np.full(4, 1234.5678901), id="large-constant"),
        ],
    )
    def test_a_constant_prediction_is_not_mistaken_for_variation(self, predictions):
        _, scores = grouped_scores(np.zeros(4, dtype=int), predictions, np.arange(4.0), ["Pearson", "Spearman"])

        assert np.isnan(scores["Pearson"][0])
        assert np.isnan(scores["Spearman"][0])


class TestMeanOverGroups:
    def test_undefined_groups_are_skipped_not_counted_as_zero(self):
        assert mean_over_groups(np.array([0.5, np.nan, 0.3])) == (pytest.approx(0.4), 2)

    @pytest.mark.parametrize("values", [np.array([np.nan, np.nan]), np.array([])])
    def test_no_defined_group_gives_nan_and_zero(self, values):
        mean, count = mean_over_groups(values)

        assert np.isnan(mean)
        assert count == 0


class TestRunGroupScore:
    def test_averages_the_defined_drugs_of_a_run(self):
        run = make_run_result(n_pairs=20, n_drugs=4)
        per_drug = [
            np.corrcoef(run.predictions[run.drug_ids == d], run.ground_truth[run.drug_ids == d])[0, 1]
            for d in np.unique(run.drug_ids)
        ]

        assert run_group_score(run, "drug", "Pearson") == pytest.approx(np.mean(per_drug))

    def test_a_run_without_a_defined_group_is_nan_and_an_unknown_grouping_is_rejected(self):
        assert np.isnan(run_group_score(make_run_result(n_pairs=1, n_drugs=1, n_samples=1), "drug", "Pearson"))
        with pytest.raises(ValueError, match="Unknown grouping"):
            run_group_score(make_run_result(), "tissue", "Pearson")  # type: ignore[arg-type]


X4 = np.array([1.0, 2.0, 3.0, 4.0])


ZEROS4 = np.zeros(4, dtype=int)


def _group_correlations(codes, n_groups, x, y, **kwargs):
    sums = PearsonSums(n_groups)
    sums.add(codes, x, y)
    return sums.correlations(**kwargs)


class TestGroupedPearson:
    def test_agrees_with_scipy_per_group_and_stays_in_the_unit_interval(self):
        rng = np.random.default_rng(0)
        codes = np.repeat([0, 1], 25)
        x = rng.normal(size=50)
        y = rng.normal(size=50)

        result = _group_correlations(codes, 2, x, y)

        for group in (0, 1):
            assert result[group] == pytest.approx(pearsonr(x[codes == group], y[codes == group])[0], abs=1e-9)
        assert np.all(
            np.abs(_group_correlations(rng.integers(0, 5, size=200), 5, np.tile(x, 4), np.tile(x, 4) * 3)) <= 1.0
        )

    def test_perfect_relationships(self):
        assert _group_correlations(ZEROS4, 1, X4, 2 * X4 + 1)[0] == pytest.approx(1.0)
        assert _group_correlations(ZEROS4, 1, X4, -X4)[0] == pytest.approx(-1.0)

    def test_degenerate_groups_are_nan(self):
        single = _group_correlations(np.array([0, 1, 1]), 2, np.array([1.0, 1.0, 2.0]), np.array([1.0, 3.0, 4.0]))
        no_variance = _group_correlations(ZEROS4, 1, np.ones(4), X4)
        empty = _group_correlations(np.zeros(3, dtype=int), 3, np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0]))
        all_dropped = _group_correlations(np.array([-1, -1]), 2, np.array([1.0, 2.0]), np.array([1.0, 2.0]))

        assert np.isnan(single[0])
        assert np.isfinite(single[1])
        assert np.isnan(no_variance[0])
        assert empty.shape == (3,)
        assert np.isnan(empty[1:]).all()
        assert np.isnan(all_dropped).all()

    def test_negative_codes_and_nan_observations_are_dropped(self):
        negative = _group_correlations(
            np.array([-1, 0, 0, 0]), 1, np.array([99.0, 1, 2, 3]), np.array([-99.0, 1, 2, 3])
        )
        nan = _group_correlations(
            np.zeros(5, dtype=int), 1, np.array([np.nan, 1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0, 2.0, 3.0, np.nan])
        )
        below_minimum = _group_correlations(
            np.zeros(3, dtype=int), 1, np.array([1.0, np.nan, np.nan]), np.array([1.0, 2, 3])
        )

        assert negative[0] == pytest.approx(1.0)
        assert nan[0] == pytest.approx(1.0)
        assert np.isnan(below_minimum[0])

    def test_a_higher_min_count_filters_small_groups(self):
        codes = np.zeros(3, dtype=int)
        x = np.array([1.0, 2.0, 3.0])

        assert np.isfinite(_group_correlations(codes, 1, x, x, min_count=3)[0])
        assert np.isnan(_group_correlations(codes, 1, x, x, min_count=4)[0])


class TestConstantDetection:
    @pytest.mark.parametrize(
        ("x", "is_nan"),
        [
            pytest.param(np.full(50, 1234.5678901), True, id="large-constant"),
            pytest.param(5.0 + np.array([0.0, CONSTANT_ATOL / 10] * 25), True, id="within-tolerance"),
            pytest.param(1e5 + np.array([0.0, 0.5] * 25), True, id="half-unit-at-1e5"),
            pytest.param(1e5 + np.array([0.0, 2.0] * 25), False, id="two-units-at-1e5"),
        ],
    )
    def test_the_constant_rule(self, x, is_nan):
        assert np.isnan(_group_correlations(np.zeros(50, dtype=int), 1, x, np.arange(50.0))[0]) is np.bool_(is_nan)

    def test_each_group_is_judged_on_its_own_range(self):
        codes = np.array([0, 0, 1, 1, 2, 2])
        x = np.array([0.0, 5e-7, 0.0, 2e-6, 1e5, 1e5 + 0.5])

        result = _group_correlations(codes, 3, x, np.array([0.0, 1.0] * 3))

        assert np.isnan(result).tolist() == [True, False, True]

    def test_an_empty_group_does_not_vary(self):
        sums = PearsonSums(2)
        sums.add(np.array([0, 0]), np.array([1.0, 2.0]), np.array([3.0, 5.0]))

        assert sums.varies().tolist() == [True, False]
        assert sums.y_varies().tolist() == [True, False]


SERIES = {
    "constant": np.full(6, 3.0),
    "zero_and_5e-7": np.array([0.0, 5e-7] * 3),
    "zero_and_2e-6": np.array([0.0, 2e-6] * 3),
    "1e5_offset_tiny_range": 1e5 + np.array([0.0, 1e-7] * 3),
    "1e5_offset_half_range": 1e5 + np.array([0.0, 0.5] * 3),
    "1e5_offset_wide_range": 1e5 + np.array([0.0, 5.0] * 3),
    "negative_1e5_offset": -1e5 + np.array([0.0, 0.5] * 3),
    "varying": np.arange(6.0),
}


@pytest.mark.parametrize("name", SERIES)
def test_grouped_and_pooled_agree_on_which_series_is_constant_in_any_order(name):
    series, varying = SERIES[name], np.arange(6.0)
    codes = np.zeros(6, dtype=int)
    rng = np.random.default_rng(0)

    assert np.isnan(_group_correlations(codes, 1, series, varying)[0]) == is_constant(series)
    assert np.isnan(_group_correlations(codes, 1, varying, series)[0]) == is_constant(series)
    assert (pearson(series, varying) == 0.0) == is_constant(series)
    assert np.isnan(pearson(varying, series)) == is_constant(series)
    for _ in range(10):
        order = rng.permutation(6)
        assert np.isnan(_group_correlations(codes, 1, series[order], varying[order])[0]) == is_constant(series)


def test_streamed_folds_equal_one_pooled_batch_and_counts_are_per_group():
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 3, size=60)
    x, y = rng.normal(size=60), rng.normal(size=60)
    streamed = PearsonSums(3)
    streamed.add(codes[:30], x[:30], y[:30])
    streamed.add(codes[30:], x[30:], y[30:])

    np.testing.assert_allclose(streamed.correlations(), _group_correlations(codes, 3, x, y))
    np.testing.assert_array_equal(streamed.counts, np.bincount(codes, minlength=3))
