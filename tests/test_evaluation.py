import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.evaluation import evaluate, pearson, spearman, kendall, partial_correlation


def test_evaluate():
    # Create mock dataset
    predictions = np.array([1, 2, 3, 4, 5])
    response = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    dataset = DrugResponseDataset(
        response=response,
        cell_line_ids=pd.Series(["A", "B", "C", "D", "E"]),
        drug_ids=pd.Series(["a", "b", "c", "d", "e"]),
        predictions=predictions,
    )

    # Test metrics calculation
    mse_expected = mean_squared_error(predictions, response)
    rmse_expected = np.sqrt(mean_squared_error(predictions, response))
    mae_expected = mean_absolute_error(predictions, response)
    r2_expected = r2_score(y_pred=predictions, y_true=response)

    # Evaluate using all available metrics
    results = evaluate(dataset, metric=["MSE", "RMSE", "MAE", "R^2"])

    # Check if the calculated metrics match the expected values
    assert np.isclose(
        results["MSE"], mse_expected
    ), f"Expected mse: {mse_expected}, Got: {results['MSE']}"
    assert np.isclose(
        results["RMSE"], rmse_expected
    ), f"Expected rmse: {rmse_expected}, Got: {results['RMSE']}"
    assert np.isclose(
        results["MAE"], mae_expected
    ), f"Expected mae: {mae_expected}, Got: {results['MAE']}"
    assert np.isclose(
        results["R^2"], r2_expected
    ), f"Expected, r2: {r2_expected}, Got: {results['R^2']}"


# Mock dataset generation function
@pytest.fixture
def generate_mock_data():
    response = np.random.rand(100) * 8
    cell_line_ids = np.random.randint(0, 10, size=100).astype(str)
    drug_ids = np.random.randint(20, 30, size=100).astype(str)
    return response, cell_line_ids, drug_ids

@pytest.fixture
def generate_mock_anticorrelated_data():
    response = np.arange(2e6, 0, -1)
    y_pred = response[::-1]
    return y_pred, response


@pytest.fixture
def generate_mock_uncorrelated_data():
    response = np.arange(2e6)
    y_pred = np.random.permutation(response)
    return y_pred, response


@pytest.fixture
def generate_mock_correlated_data():
    response = np.arange(2e6)
    y_pred = response
    return y_pred, response


def test_partial_correlation(generate_mock_data):
    response, cell_line_ids, drug_ids = generate_mock_data

    df = pd.DataFrame({
        "response": response,
        "cell_line_id": cell_line_ids,
        "drug_id": drug_ids
    })

    df["mean_per_drug"] = df.groupby("drug_id")["response"].transform("mean")
    df["mean_per_cell_line"] = df.groupby("cell_line_id")["response"].transform("mean")

    y_pred = np.array(df["mean_per_drug"])
    # add gaussian noise to y_pred
    y_pred += np.random.normal(0, 0.1, size=len(y_pred))

    pc = partial_correlation(y_pred, response, cell_line_ids, drug_ids)
    assert np.isclose(pc, 0.0, atol=1e-3)

    y_pred = np.array(df["mean_per_cell_line"])
    # add gaussian noise to y_pred
    y_pred += np.random.normal(0, 0.1, size=len(y_pred))

    pc = partial_correlation(y_pred, response, cell_line_ids, drug_ids)
    assert np.isclose(pc, 0.0, atol=1e-3)



def test_pearson_correlated(generate_mock_correlated_data):
    y_pred, response = generate_mock_correlated_data

    pc = pearson(y_pred, response)
    assert np.isclose(pc, 1.0, atol=1e-3)


def test_pearson_anticorrelated(generate_mock_anticorrelated_data):
    y_pred, response = generate_mock_anticorrelated_data

    pc = pearson(y_pred, response)
    assert np.isclose(pc, -1.0, atol=1e-1)


@flaky(max_runs=3)
def test_pearson_uncorrelated(generate_mock_uncorrelated_data):
    y_pred, response = generate_mock_uncorrelated_data

    pc = pearson(y_pred, response)
    assert np.isclose(pc, 0.0, atol=1e-3)


def test_spearman_correlated(generate_mock_correlated_data):
    y_pred, response = generate_mock_correlated_data

    sp = spearman(y_pred, response)
    assert np.isclose(sp, 1.0, atol=1e-3)


def test_spearman_anticorrelated(generate_mock_anticorrelated_data):
    y_pred, response = generate_mock_anticorrelated_data

    sp = spearman(y_pred, response)
    assert np.isclose(sp, -1.0, atol=1e-1)


@flaky(max_runs=3)
def test_spearman_uncorrelated(generate_mock_uncorrelated_data):
    y_pred, response = generate_mock_uncorrelated_data

    sp = spearman(y_pred, response)
    print(sp)
    assert np.isclose(sp, 0.0, atol=1e-3)


def test_kendall_correlated(generate_mock_correlated_data):
    y_pred, response = generate_mock_correlated_data

    kd = kendall(y_pred, response)
    assert np.isclose(kd, 1.0, atol=1e-3)


def test_kendall_anticorrelated(generate_mock_anticorrelated_data):
    y_pred, response = generate_mock_anticorrelated_data

    kd = kendall(y_pred, response)
    assert np.isclose(kd, -1.0, atol=1e-1)


@flaky(max_runs=3)
def test_kendall_uncorrelated(generate_mock_uncorrelated_data):
    y_pred, response = generate_mock_uncorrelated_data

    kd = kendall(y_pred, response)
    assert np.isclose(kd, 0.0, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])