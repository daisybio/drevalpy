"""Tests for evaluation.py."""

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from drevalpy.evaluation import evaluate, is_constant, kendall, pearson, score_predictions, spearman

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
