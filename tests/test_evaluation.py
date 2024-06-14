import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.evaluation import evaluate


def test_evaluate():
    # Create mock dataset
    predictions = np.array([1, 2, 3, 4, 5])
    response = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    dataset = DrugResponseDataset(predictions=predictions, response=response)

    # Test metrics calculation
    mse_expected = mean_squared_error(predictions, response)
    rmse_expected = np.sqrt(mean_squared_error(predictions, response))
    mae_expected = mean_absolute_error(predictions, response)
    r2_expected = r2_score(y_pred=predictions, y_true=response)

    # Evaluate using all available metrics
    results = evaluate(dataset, metric=["mse", "rmse", "mae", "r2"])

    # Check if the calculated metrics match the expected values
    assert np.isclose(
        results["mse"], mse_expected
    ), f"Expected mse: {mse_expected}, Got: {results['mse']}"
    assert np.isclose(
        results["rmse"], rmse_expected
    ), f"Expected rmse: {rmse_expected}, Got: {results['rmse']}"
    assert np.isclose(
        results["mae"], mae_expected
    ), f"Expected mae: {mae_expected}, Got: {results['mae']}"
    assert np.isclose(
        results["r2"], r2_expected
    ), f"Expected, r2: {r2_expected}, Got: {results['r2']}"
