"""Compatibility tests for legacy drevalpy.models import paths."""

from __future__ import annotations

import importlib

import pytest

from drevalpy.models import MODEL_FACTORY

_LEGACY_MODULE_PATHS = (
    "drevalpy.models.DIPK.dipk",
    "drevalpy.models.DIPK.data_utils",
    "drevalpy.models.DIPK.model_utils",
    "drevalpy.models.DIPK.attention_utils",
    "drevalpy.models.DIPK.gene_expression_encoder",
    "drevalpy.models.DrugGNN.drug_gnn",
    "drevalpy.models.MOLIR.molir",
    "drevalpy.models.MOLIR.utils",
    "drevalpy.models.PharmaFormer.pharmaformer",
    "drevalpy.models.PharmaFormer.model_utils",
    "drevalpy.models.Precily.precily",
    "drevalpy.models.Precily.model_utils",
    "drevalpy.models.SRMF.srmf",
    "drevalpy.models.SimpleNeuralNetwork.simple_neural_network",
    "drevalpy.models.SimpleNeuralNetwork.multi_view_neural_network",
    "drevalpy.models.SimpleNeuralNetwork.utils",
    "drevalpy.models.SuperFELTR.superfeltr",
    "drevalpy.models.SuperFELTR.utils",
    "drevalpy.models.baselines.sklearn_models",
    "drevalpy.models.baselines.naive_pred",
    "drevalpy.models.baselines.multi_view_random_forest",
    "drevalpy.models.baselines.multi_view_xgboost",
    "drevalpy.models.baselines.multi_view_lightgbm",
    "drevalpy.models.baselines.singledrug_baselines",
)

_LEGACY_SYMBOLS = {
    "drevalpy.models.DIPK.dipk": ("DIPKModel",),
    "drevalpy.models.DIPK.data_utils": ("load_bionic_features", "get_data", "CollateFn", "DIPKDataset"),
    "drevalpy.models.DIPK.model_utils": ("AttentionLayer", "DenseLayers", "Predictor"),
    "drevalpy.models.DIPK.attention_utils": ("MultiHeadAttentionLayer",),
    "drevalpy.models.DIPK.gene_expression_encoder": (
        "GeneExpressionEncoder",
        "GeneExpressionDecoder",
        "train_gene_expession_autoencoder",
    ),
    "drevalpy.models.DrugGNN.drug_gnn": ("DrugGNN",),
    "drevalpy.models.MOLIR.molir": ("MOLIR",),
    "drevalpy.models.MOLIR.utils": ("MOLIModel", "MOLIEncoder"),
    "drevalpy.models.PharmaFormer.pharmaformer": ("PharmaFormerModel",),
    "drevalpy.models.PharmaFormer.model_utils": ("CombinedModel", "TransModel"),
    "drevalpy.models.Precily.precily": ("PrecilyModel",),
    "drevalpy.models.Precily.model_utils": ("PrecilyNetwork",),
    "drevalpy.models.SRMF.srmf": ("SRMF",),
    "drevalpy.models.SimpleNeuralNetwork.simple_neural_network": ("SimpleNeuralNetwork",),
    "drevalpy.models.SimpleNeuralNetwork.multi_view_neural_network": ("MultiViewNeuralNetwork",),
    "drevalpy.models.SimpleNeuralNetwork.utils": ("FeedForwardNetwork",),
    "drevalpy.models.SuperFELTR.superfeltr": ("SuperFELTR",),
    "drevalpy.models.SuperFELTR.utils": ("SuperFELTEncoder", "SuperFELTRegressor"),
    "drevalpy.models.baselines.sklearn_models": ("ElasticNetModel", "RandomForest", "SklearnModel"),
    "drevalpy.models.baselines.naive_pred": ("NaivePredictor", "NaiveDrugMeanPredictor"),
    "drevalpy.models.baselines.multi_view_random_forest": ("MultiViewRandomForest",),
    "drevalpy.models.baselines.multi_view_xgboost": ("MultiViewXGBoost",),
    "drevalpy.models.baselines.multi_view_lightgbm": ("MultiViewLightGBM",),
    "drevalpy.models.baselines.singledrug_baselines": ("SingleDrugElasticNet", "SingleDrugRandomForest"),
}


@pytest.mark.parametrize("module_path", _LEGACY_MODULE_PATHS)
def test_legacy_model_module_imports(module_path: str) -> None:
    module = importlib.import_module(module_path)
    for symbol in _LEGACY_SYMBOLS[module_path]:
        assert hasattr(module, symbol), f"{module_path} missing {symbol}"


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORY))
def test_model_factory_models_instantiate(model_name: str) -> None:
    model = MODEL_FACTORY[model_name]()
    assert model.get_model_name() == model_name
    hyperparameters = model.get_hyperparameter_set()[0]
    if model_name == "DIPK":
        hyperparameters = {**hyperparameters, "epochs": 1, "epochs_autoencoder": 1, "heads": 1}
    elif model_name in {"SimpleNeuralNetwork", "MultiViewNeuralNetwork"}:
        hyperparameters = {**hyperparameters, "units_per_layer": [2, 2], "max_epochs": 1}
    elif model_name == "PharmaFormer":
        hyperparameters = {**hyperparameters, "epochs": 1, "patience": 2}
    elif model_name == "Precily":
        hyperparameters = {**hyperparameters, "epochs": 1, "batch_size": 32}
    try:
        model.build_model(hyperparameters=hyperparameters)
    except ImportError as exc:
        if model_name in {"MultiViewXGBoost", "MultiViewLightGBM"}:
            pytest.skip(str(exc))
        raise
