"""Tests for evaluation.py."""

import numpy as np
import pytest
from flaky import flaky
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from drevalpy.evaluation import evaluate, kendall, pearson, score_predictions, spearman


def test_evaluate() -> None:
    """Test the evaluate function."""
    predictions = np.array([1, 2, 3, 4, 5])
    response = np.array([1.1, 2.2, 3.3, 4.4, 5.5])

    # Test metrics calculation
    mse_expected = mean_squared_error(predictions, response)
    rmse_expected = np.sqrt(mean_squared_error(predictions, response))
    mae_expected = mean_absolute_error(predictions, response)
    r2_expected = r2_score(y_pred=predictions, y_true=response)

    # Evaluate using all available metrics
    results = evaluate(predictions, response, metric=["MSE", "RMSE", "MAE", "R^2"])

    # Check if the calculated metrics match the expected values
    assert np.isclose(results["MSE"], mse_expected), f"Expected mse: {mse_expected}, Got: {results['MSE']}"
    assert np.isclose(results["RMSE"], rmse_expected), f"Expected rmse: {rmse_expected}, Got: {results['RMSE']}"
    assert np.isclose(results["MAE"], mae_expected), f"Expected mae: {mae_expected}, Got: {results['MAE']}"
    assert np.isclose(results["R^2"], r2_expected), f"Expected, r2: {r2_expected}, Got: {results['R^2']}"


# Mock dataset generation function
@pytest.fixture
def generate_mock_data_drug_mean() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate mock data with a mean response per drug.

    :returns: response, cell_line_ids, drug_ids
    """
    response_list = []
    drug_ids = []
    cell_line_ids = []
    for drug in [f"drug_{i}" for i in range(100)]:
        drug_mean = np.random.randint(0, 8)
        for cell_line in [f"cell_line_{i}" for i in range(200)]:
            response = np.random.normal(drug_mean, 0.1)
            response_list.append(response)
            drug_ids.append(drug)
            cell_line_ids.append(cell_line)
    return np.array(response_list), np.array(cell_line_ids), np.array(drug_ids)


@pytest.fixture
def generate_mock_data_constant_prediction() -> tuple[np.ndarray, np.ndarray]:
    """Generate mock data with constant prediction.

    :returns: y_pred, response
    """
    response = np.arange(2e6)
    y_pred = np.ones_like(response, dtype=float)
    return y_pred, response


@pytest.fixture
def generate_mock_anticorrelated_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate mock data with anticorrelated prediction.

    :returns: y_pred, response
    """
    response = np.arange(2e6, 0, -1)
    y_pred = response[::-1]
    return y_pred, response


@pytest.fixture
def generate_mock_uncorrelated_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate mock data with uncorrelated prediction.

    Uses a fixed RNG seed so correlation asserts stay deterministic across platforms.

    :returns: y_pred, response
    """
    response = np.arange(2e6)
    y_pred = np.random.default_rng(0).permutation(response)
    return y_pred, response


@pytest.fixture
def generate_mock_correlated_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate mock data with correlated prediction.

    :returns: y_pred, response
    """
    response = np.arange(2e6)
    y_pred = response
    return y_pred, response


def test_pearson_correlated(generate_mock_correlated_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the pearson correlation function.

    :param generate_mock_correlated_data: mock data generator
    """
    y_pred, response = generate_mock_correlated_data

    pc = pearson(y_pred, response)
    assert np.isclose(pc, 1.0, atol=1e-3)


def test_pearson_anticorrelated(generate_mock_anticorrelated_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the pearson correlation function.

    :param generate_mock_anticorrelated_data: mock data generator
    """
    y_pred, response = generate_mock_anticorrelated_data

    pc = pearson(y_pred, response)
    assert np.isclose(pc, -1.0, atol=1e-1)


@flaky(max_runs=3)
def test_pearson_uncorrelated(generate_mock_uncorrelated_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the pearson correlation function.

    :param generate_mock_uncorrelated_data: mock data generator
    """
    y_pred, response = generate_mock_uncorrelated_data

    pc = pearson(y_pred, response)
    assert np.isclose(pc, 0.0, atol=1e-3)


def test_spearman_correlated(generate_mock_correlated_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the spearman correlation function.

    :param generate_mock_correlated_data: mock data generator
    """
    y_pred, response = generate_mock_correlated_data

    sp = spearman(y_pred, response)
    assert np.isclose(sp, 1.0, atol=1e-3)


def test_spearman_anticorrelated(generate_mock_anticorrelated_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the spearman correlation function.

    :param generate_mock_anticorrelated_data: mock data generator
    """
    y_pred, response = generate_mock_anticorrelated_data

    sp = spearman(y_pred, response)
    assert np.isclose(sp, -1.0, atol=1e-1)


@flaky(max_runs=3)
def test_spearman_uncorrelated(generate_mock_uncorrelated_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the spearman correlation function.

    :param generate_mock_uncorrelated_data: mock data generator
    """
    y_pred, response = generate_mock_uncorrelated_data

    sp = spearman(y_pred, response)
    assert np.isclose(sp, 0.0, atol=1e-3)


def test_kendall_correlated(generate_mock_correlated_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the kendall correlation function.

    :param generate_mock_correlated_data: mock data generator
    """
    y_pred, response = generate_mock_correlated_data

    kd = kendall(y_pred, response)
    assert np.isclose(kd, 1.0, atol=1e-3)


def test_kendall_anticorrelated(generate_mock_anticorrelated_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the kendall correlation function.

    :param generate_mock_anticorrelated_data: mock data generator
    """
    y_pred, response = generate_mock_anticorrelated_data

    kd = kendall(y_pred, response)
    assert np.isclose(kd, -1.0, atol=1e-1)


@flaky(max_runs=3)
def test_kendall_uncorrelated(generate_mock_uncorrelated_data):
    """Test the kendall correlation function.

    :param generate_mock_uncorrelated_data: mock data generator
    """
    y_pred, response = generate_mock_uncorrelated_data

    kd = kendall(y_pred, response)
    assert np.isclose(kd, 0.0, atol=1e-3)


def test_correlations_constant_prediction(
    generate_mock_data_constant_prediction: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test the correlation functions with constant prediction.

    :param generate_mock_data_constant_prediction: mock data generator
    """
    y_pred, response = generate_mock_data_constant_prediction
    pc = pearson(y_pred, response)
    sp = spearman(y_pred, response)
    kd = kendall(y_pred, response)
    assert np.isclose(pc, 0.0, atol=1e-3)
    assert np.isclose(sp, 0.0, atol=1e-3)
    assert np.isclose(kd, 0.0, atol=1e-3)


def test_score_predictions_all_nan_predictions() -> None:
    response = np.array([1.0, 2.0])
    predictions = np.array([np.nan, np.nan])
    assert np.isnan(score_predictions(predictions, response, ["RMSE"])["RMSE"])


@pytest.mark.parametrize("correlation", [pearson, spearman, kendall])
def test_correlations_raise_value_error_on_length_mismatch(correlation) -> None:
    with pytest.raises(ValueError, match="same length"):
        correlation(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_score_predictions_raises_value_error_for_an_unknown_metric() -> None:
    with pytest.raises(ValueError, match="invalid metric"):
        score_predictions(np.array([1.0, 2.0]), np.array([1.0, 2.0]), ["nope"])


def test_evaluate_raises_value_error_for_an_unknown_metric() -> None:
    with pytest.raises(ValueError, match="invalid metric"):
        evaluate(np.array([1.0, 2.0]), np.array([1.0, 2.0]), metric="nope")


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="nothing"),
        pytest.param({"predictions": np.array([1.0, 2.0])}, id="keyword-predictions-without-response"),
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
    from drevalpy.evaluation import score_predictions

    assert score_predictions(np.array([1.5, np.nan, 2.0]), np.array([4.0, 1.0, np.nan]), [metric])[metric] == expected


@pytest.mark.parametrize("metric", ["R^2", "Pearson", "Spearman", "Kendall"])
def test_statistical_metrics_require_two_jointly_valid_pairs(metric):
    from drevalpy.evaluation import score_predictions

    assert np.isnan(score_predictions(np.array([1.5]), np.array([4.0]), [metric])[metric])


@pytest.mark.parametrize("predictions, targets", [(np.ones(2), np.ones(3)), (np.ones((2, 1)), np.ones(2))])
def test_scoring_rejects_alignment_errors(predictions, targets):
    from drevalpy.evaluation import score_predictions

    with pytest.raises(ValueError, match="equal-shaped one-dimensional"):
        score_predictions(predictions, targets)


def test_empty_scoring_conventions_share_the_same_mask():
    from drevalpy.evaluation import score_predictions

    predictions, targets = np.array([np.nan, 2.0]), np.array([1.0, np.nan])
    assert score_predictions(predictions, targets, empty_as_dict=True) == {}
    assert all(np.isnan(v) for v in score_predictions(predictions, targets).values())
