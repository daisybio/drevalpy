"""Tests for the DRPModel."""

from drevalpy.models._model_lookup import known_model_names


def test_factory() -> None:
    """Test that known model names include the built-in zoo presets."""
    names = set(known_model_names(include_external=False))
    assert "NaivePredictor" in names
    assert "NaiveDrugMeanPredictor" in names
    assert "NaiveCellLineMeanPredictor" in names
    assert "NaiveMeanEffectsPredictor" in names
    assert "NaiveTissueDrugMeanPredictor" in names
    assert "ElasticNet" in names
    assert "RandomForest" in names
    assert "SVR" in names
    assert "SimpleNeuralNetwork" in names
    assert "MultiViewNeuralNetwork" in names
    assert "MultiViewRandomForest" in names
    assert "SingleDrugRandomForest" in names
    assert "SRMF" in names
    assert "GradientBoosting" in names
    assert "MOLIR" in names
    assert "SuperFELTR" in names
    assert "DIPK" in names
    assert "SparseGO" in names
