"""Tests for ``drevalpy.evaluation._metrics``: ``Metric``, ``LibraryFunction`` and the undefined-score rule."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import drevalpy.evaluation
from drevalpy.evaluation import Metric, MetricKind, evaluate, evaluate_per_group
from drevalpy.evaluation._metrics import MIN_GROUP_SIZE, LibraryFunction, score_pairs
from drevalpy.evaluation._pairs import CONSTANT_ATOL, Pairs, group_pairs, is_constant

CORRELATIONS = [Metric.PEARSON, Metric.SPEARMAN, Metric.KENDALL]
ERRORS = [Metric.MSE, Metric.RMSE, Metric.MAE]
FIT_AND_CORRELATIONS = [Metric.R2, *CORRELATIONS]

REFERENCES = {
    Metric.MSE: lambda pred, true: mean_squared_error(true, pred),
    Metric.RMSE: lambda pred, true: np.sqrt(mean_squared_error(true, pred)),
    Metric.MAE: lambda pred, true: mean_absolute_error(true, pred),
    Metric.R2: lambda pred, true: r2_score(true, pred),
    Metric.PEARSON: lambda pred, true: pearsonr(pred, true)[0],
    Metric.SPEARMAN: lambda pred, true: spearmanr(pred, true)[0],
    Metric.KENDALL: lambda pred, true: kendalltau(pred, true)[0],
}

N = 50_000
RESPONSE = np.arange(float(N))


def _score(metric: Metric, y_pred, y_true) -> float:
    return score_pairs(metric, Pairs.finite(y_pred, y_true))


def _scores_by_group(metric: Metric, y_pred, y_true, labels, groups=None) -> np.ndarray:
    return evaluate_per_group(y_pred, y_true, metric, labels=labels, groups=groups).values[metric.value]


@pytest.fixture(scope="module")
def grouped_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels, predictions and response of six groups of unequal size, with ties in both variables."""
    rng = np.random.default_rng(3)
    labels = np.repeat([f"D{i}" for i in range(6)], [5, 8, 15, 22, 40, 11])
    response = np.round(rng.normal(size=labels.size), 1)
    predictions = np.round(response + rng.normal(size=labels.size), 1)
    return labels, predictions, response


class TestMetric:
    def test_a_metric_is_its_stored_name(self) -> None:
        assert Metric.PEARSON == "Pearson"
        assert Metric.R2 == "R^2"
        assert Metric.R2.value == "R^2"
        assert {"Pearson": 1}[Metric.PEARSON] == 1
        assert f"{Metric.SPEARMAN}" == "Spearman"

    def test_metrics_are_listed_in_report_order(self) -> None:
        assert [str(metric) for metric in Metric] == ["MSE", "RMSE", "MAE", "R^2", "Pearson", "Spearman", "Kendall"]

    def test_a_name_looks_up_its_member(self) -> None:
        assert Metric("R^2") is Metric.R2
        assert all(Metric(metric.value) is metric for metric in Metric)

    @pytest.mark.parametrize("name", ["Nope", "pearson", "R2"])
    def test_an_unknown_name_lists_the_available_ones(self, name) -> None:
        with pytest.raises(ValueError, match=f"Unknown metric '{name}'. Available: MSE, RMSE, MAE, R\\^2, Pearson"):
            Metric(name)

    @pytest.mark.parametrize(
        ("metric", "kind", "direction"),
        [
            (Metric.MSE, MetricKind.ERROR, "min"),
            (Metric.RMSE, MetricKind.ERROR, "min"),
            (Metric.MAE, MetricKind.ERROR, "min"),
            (Metric.R2, MetricKind.FIT, "max"),
            (Metric.PEARSON, MetricKind.CORRELATION, "max"),
            (Metric.SPEARMAN, MetricKind.CORRELATION, "max"),
            (Metric.KENDALL, MetricKind.CORRELATION, "max"),
        ],
    )
    def test_kind_and_direction(self, metric, kind, direction) -> None:
        assert metric.kind is kind
        assert metric.direction == direction

    @pytest.mark.parametrize(
        ("metric", "path"),
        [
            (Metric.MSE, "sklearn.metrics.mean_squared_error"),
            (Metric.RMSE, "sklearn.metrics.root_mean_squared_error"),
            (Metric.MAE, "sklearn.metrics.mean_absolute_error"),
            (Metric.R2, "sklearn.metrics.r2_score"),
            (Metric.PEARSON, "scipy.stats.pearsonr"),
            (Metric.SPEARMAN, "scipy.stats.spearmanr"),
            (Metric.KENDALL, "scipy.stats.kendalltau"),
        ],
    )
    def test_library_function_path(self, metric, path) -> None:
        assert metric.function.path == path

    def test_a_library_function_takes_predictions_before_response_for_every_library(self) -> None:
        y_pred, y_true = np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 4.0])

        assert Metric.MSE.function(y_pred, y_true) == pytest.approx(1 / 3)
        assert Metric.R2.function(y_pred, y_true) == pytest.approx(r2_score(y_true, y_pred))
        assert Metric.PEARSON.function(y_pred, y_true) == pytest.approx(pearsonr(y_pred, y_true)[0])

    def test_a_library_function_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            Metric.MSE.function.name = "other"  # type: ignore[misc]

    def test_a_library_function_is_a_value(self) -> None:
        assert LibraryFunction("scipy.stats", "pearsonr") == Metric.PEARSON.function

    def test_the_public_names_of_the_package(self) -> None:
        expected = {"GroupBy", "Metric", "MetricKind", "PerGroupScores", "evaluate", "evaluate_per_group"}

        assert set(drevalpy.evaluation.__all__) == expected
        assert len(drevalpy.evaluation.__all__) == len(expected)
        assert all(hasattr(drevalpy.evaluation, name) for name in expected)


class TestAgainstScipyAndSklearn:
    @pytest.mark.parametrize("offset", [0.0, 1e3], ids=["random", "offset-1e3"])
    @pytest.mark.parametrize("metric", Metric)
    def test_pooled_and_per_group_scores_match_the_reference(self, grouped_data, metric, offset) -> None:
        labels, predictions, response = grouped_data
        predictions, response = predictions + offset, response + offset

        scores = _scores_by_group(metric, predictions, response, labels)

        for index in range(6):
            mask = labels == f"D{index}"
            expected = REFERENCES[metric](predictions[mask], response[mask])
            assert scores[index] == pytest.approx(expected, rel=1e-9, abs=1e-12)
            assert _score(metric, predictions[mask], response[mask]) == pytest.approx(expected, rel=1e-9, abs=1e-12)
        assert _score(metric, predictions, response) == pytest.approx(
            REFERENCES[metric](predictions, response), rel=1e-9, abs=1e-12
        )
        assert evaluate(predictions, response, metric)[metric.value] == _score(metric, predictions, response)

    def test_the_reference_table_covers_every_metric(self) -> None:
        assert set(REFERENCES) == set(Metric)


class TestUndefinedScores:
    """The split rule: the data decides ``NaN``, a constant prediction decides ``0.0`` for a correlation."""

    @pytest.mark.parametrize("metric", Metric)
    def test_no_pair_is_undefined_for_every_metric(self, metric) -> None:
        assert np.isnan(_score(metric, [], []))
        assert np.isnan(_score(metric, [np.nan], [1.0]))

    @pytest.mark.parametrize("metric, expected", [(Metric.MSE, 6.25), (Metric.RMSE, 2.5), (Metric.MAE, 2.5)])
    def test_error_metrics_score_one_pair(self, metric, expected) -> None:
        assert _score(metric, [1.5], [4.0]) == expected

    @pytest.mark.parametrize("metric", FIT_AND_CORRELATIONS)
    def test_fit_and_correlations_need_two_pairs(self, metric) -> None:
        assert MIN_GROUP_SIZE == 2
        assert np.isnan(_score(metric, [1.5], [4.0]))
        assert np.isfinite(_score(metric, [1.0, 2.0], [1.0, 3.0]))

    def test_two_pairs_are_scored(self) -> None:
        assert _score(Metric.R2, [1.0, 2.0], [1.0, 3.0]) == pytest.approx(0.5)
        assert [_score(metric, [1.0, 2.0], [1.0, 3.0]) for metric in CORRELATIONS] == pytest.approx([1.0, 1.0, 1.0])

    @pytest.mark.parametrize(
        "predictions",
        [
            pytest.param([3.0, 1.0, 4.0, 2.0], id="varying-prediction"),
            pytest.param([2.0] * 4, id="constant-prediction"),
        ],
    )
    @pytest.mark.parametrize("metric", FIT_AND_CORRELATIONS)
    def test_a_constant_response_is_undefined(self, metric, predictions) -> None:
        assert np.isnan(_score(metric, predictions, [5.0] * 4))

    @pytest.mark.parametrize("metric", ERRORS)
    def test_error_metrics_stay_defined_for_a_constant_response(self, metric) -> None:
        assert _score(metric, [3.0, 1.0, 4.0, 2.0], [5.0] * 4) > 0

    @pytest.mark.parametrize(
        "predictions",
        [
            pytest.param([2.0] * 4, id="constant"),
            pytest.param(5.0 + np.array([0.0, CONSTANT_ATOL / 10, 0.0, CONSTANT_ATOL / 10]), id="nearly-constant"),
            pytest.param(np.full(4, 1234.5678901), id="large-constant"),
        ],
    )
    @pytest.mark.parametrize("metric", CORRELATIONS)
    def test_a_constant_prediction_scores_zero_for_a_correlation(self, metric, predictions) -> None:
        assert _score(metric, predictions, [1.0, 2.0, 3.0, 4.0]) == 0.0

    def test_a_constant_prediction_keeps_r2_by_the_formula(self) -> None:
        response = np.array([1.0, 2.0, 3.0, 4.0])

        assert _score(Metric.R2, [2.0] * 4, response) == pytest.approx(1 - 6 / 5)
        assert _score(Metric.R2, [2.0] * 4, response) == pytest.approx(r2_score(response, [2.0] * 4))
        assert _score(Metric.R2, np.full(4, response.mean()), response) == pytest.approx(0.0)

    @pytest.mark.parametrize("metric", FIT_AND_CORRELATIONS)
    def test_both_sides_constant_is_undefined(self, metric) -> None:
        assert np.isnan(_score(metric, [2.0] * 3, [5.0] * 3))

    @pytest.mark.parametrize(
        ("predictions", "response"),
        [
            pytest.param([3.0, 1.0, 4.0, 2.0], [5.0] * 4, id="constant-response"),
            pytest.param([2.0] * 4, [1.0, 2.0, 3.0, 4.0], id="constant-prediction"),
            pytest.param([2.0] * 4, [5.0] * 4, id="both-constant"),
            pytest.param([3.0, 1.0, 4.0, 2.0], [1.0, 2.0, 4.0, 3.0], id="neither-constant"),
        ],
    )
    @pytest.mark.parametrize("metric", Metric)
    def test_the_verdict_does_not_depend_on_the_order_of_the_pairs(self, metric, predictions, response) -> None:
        predictions, response = np.array(predictions), np.array(response)
        expected = _score(metric, predictions, response)
        for seed in range(10):
            order = np.random.default_rng(seed).permutation(len(predictions))
            np.testing.assert_allclose(_score(metric, predictions[order], response[order]), expected, atol=1e-12)

    @pytest.mark.parametrize(
        "series",
        [
            pytest.param(np.full(6, 3.0), id="constant"),
            pytest.param(np.array([0.0, 5e-7] * 3), id="zero-and-5e-7"),
            pytest.param(np.array([0.0, 2e-6] * 3), id="zero-and-2e-6"),
            pytest.param(1e5 + np.array([0.0, 1e-7] * 3), id="1e5-offset-tiny-range"),
            pytest.param(1e5 + np.array([0.0, 0.5] * 3), id="1e5-offset-half-range"),
            pytest.param(1e5 + np.array([0.0, 5.0] * 3), id="1e5-offset-wide-range"),
            pytest.param(-1e5 + np.array([0.0, 0.5] * 3), id="negative-1e5-offset"),
            pytest.param(np.arange(6.0), id="varying"),
        ],
    )
    @pytest.mark.parametrize("metric", CORRELATIONS)
    def test_pooled_and_per_group_scores_agree_on_which_series_is_constant(self, metric, series) -> None:
        varying = np.arange(6.0)
        constant = is_constant(series)
        rng = np.random.default_rng(0)

        for _ in range(10):
            order = rng.permutation(6)
            assert (_score(metric, series[order], varying[order]) == 0.0) == constant
            assert np.isnan(_score(metric, varying[order], series[order])) == constant

        labels = np.repeat([0, 1], 6)
        grouped_prediction = _scores_by_group(metric, np.concatenate([series, varying]), np.tile(varying, 2), labels)
        grouped_response = _scores_by_group(metric, np.tile(varying, 2), np.concatenate([series, varying]), labels)
        assert grouped_prediction[0] == pytest.approx(_score(metric, series, varying), abs=1e-12)
        assert (grouped_prediction[0] == 0.0) == constant
        assert np.isnan(grouped_response[0]) == constant
        assert grouped_prediction[1] == pytest.approx(1.0)
        assert grouped_response[1] == pytest.approx(1.0)

    def test_a_group_is_judged_on_its_own_range(self) -> None:
        y_pred = np.array([0.0, 5e-7, 0.0, 2e-6, 1e5, 1e5 + 0.5])

        scores = _scores_by_group(Metric.PEARSON, y_pred, np.array([0.0, 1.0] * 3), np.repeat([0, 1, 2], 2))

        assert scores.tolist() == [0.0, 1.0, 0.0]

    @pytest.mark.parametrize("metric", Metric)
    def test_an_empty_group_on_a_fixed_axis_is_nan(self, metric) -> None:
        _, per_group = group_pairs(
            np.array(["A", "A"]), np.array([1.0, 2.0]), np.array([1.0, 3.0]), groups=np.array(["A", "B"])
        )

        assert np.isfinite(score_pairs(metric, per_group[0]))
        assert np.isnan(score_pairs(metric, per_group[1]))

    @pytest.mark.parametrize("metric", CORRELATIONS)
    def test_an_empty_group_on_a_fixed_axis_is_nan_not_zero(self, metric) -> None:
        scores = _scores_by_group(
            metric,
            np.array([1.0, 1.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 1.0, 2.0]),
            np.array([0, 0, 1, 1]),
            groups=np.arange(3),
        )

        assert scores[0] == 0.0
        assert scores[1] == pytest.approx(1.0)
        assert np.isnan(scores[2])

    @pytest.mark.parametrize("metric", Metric)
    def test_scoring_degenerate_input_emits_no_warning(self, metric) -> None:
        cases = [([], []), ([1.0], [2.0]), ([2.0] * 4, [1.0, 2.0, 3.0, 4.0]), ([1.0, 2.0, 3.0], [5.0] * 3)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for y_pred, y_true in cases:
                _score(metric, y_pred, y_true)
            _scores_by_group(metric, [1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 5.0, 5.0], [0, 0, 1, 1], groups=np.arange(3))


class TestScorePairs:
    @pytest.mark.parametrize("metric", CORRELATIONS)
    def test_correlation_extremes(self, metric) -> None:
        uncorrelated = np.random.default_rng(0).permutation(RESPONSE)

        assert _score(metric, RESPONSE, RESPONSE) == pytest.approx(1.0, abs=1e-3)
        assert _score(metric, RESPONSE[::-1], RESPONSE) == pytest.approx(-1.0, abs=1e-3)
        assert _score(metric, uncorrelated, RESPONSE) == pytest.approx(0.0, abs=2e-2)
        assert _score(metric, np.ones(N), RESPONSE) == 0.0

    @pytest.mark.parametrize("metric", Metric)
    def test_evaluate_raises_value_error_on_length_mismatch(self, metric) -> None:
        with pytest.raises(ValueError, match="equally long"):
            evaluate(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), metric)
