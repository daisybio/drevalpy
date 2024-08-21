import pytest

from drevalpy.models import MODEL_FACTORY


def test_factory():
    assert "SimpleNeuralNetwork" in MODEL_FACTORY
    assert "MultiOmicsNeuralNetwork" in MODEL_FACTORY
    assert "ElasticNet" in MODEL_FACTORY
    assert "RandomForest" in MODEL_FACTORY
    assert "MultiOmicsRandomForest" in MODEL_FACTORY
    assert "SVR" in MODEL_FACTORY
    assert "NaivePredictor" in MODEL_FACTORY
    assert "NaiveDrugMeanPredictor" in MODEL_FACTORY
    assert "NaiveCellLineMeanPredictor" in MODEL_FACTORY
    assert "SingleDrugRandomForest" in MODEL_FACTORY
    assert len(MODEL_FACTORY) == 10


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
