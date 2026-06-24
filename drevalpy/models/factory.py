"""Build `~drevalpy.components.config.ModelConfig` objects for existing models."""

from __future__ import annotations

from typing import Any

from drevalpy.components.config import FeaturizerConfig, ModelConfig, PredictorConfig
from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.models.featurizer_mapping import cell_line_featurizer_from_views


def _get_view_as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _views(hyperparameters: dict[str, Any], key: str, default: list[str]) -> list[str]:
    return _get_view_as_list(hyperparameters.get(key, default))


def featurizer_configs_from_view_hyperparameters(
    hyperparameters: dict[str, Any],
) -> tuple[FeaturizerConfig | None, FeaturizerConfig | None]:
    """Build featurizer configs when legacy view hyperparameters are explicitly set."""
    cell_line_featurizer = None
    drug_featurizer = None

    if "cell_line_views" in hyperparameters:
        cell_line_views = _views(hyperparameters, "cell_line_views", ["gene_expression"])
        cell_line_featurizer = cell_line_featurizer_from_views(cell_line_views, hyperparameters)

    if "drug_views" in hyperparameters:
        drug_views = _views(hyperparameters, "drug_views", ["fingerprints"])
        if drug_views:
            drug_name = "fingerprints" if drug_views[0] == "fingerprints" else drug_views[0]
            drug_featurizer = FeaturizerConfig.model_validate(
                normalize_featurizer_config(drug_name, default_registry="drug")
            )

    return cell_line_featurizer, drug_featurizer


def sklearn_model_config(predictor_type: str, hyperparameters: dict[str, Any]) -> ModelConfig:
    """Map sklearn baseline hyperparameters to a modular config."""
    cell_line_views = _views(hyperparameters, "cell_line_views", ["gene_expression"])
    drug_views = _views(hyperparameters, "drug_views", ["fingerprints"])
    cell_line_featurizer = cell_line_featurizer_from_views(cell_line_views, hyperparameters)

    drug_featurizer = None
    if drug_views:
        drug_name = "fingerprints" if drug_views[0] == "fingerprints" else drug_views[0]
        drug_featurizer = FeaturizerConfig.model_validate(
            normalize_featurizer_config(drug_name, default_registry="drug")
        )

    predictor_hp = {
        key: value for key, value in hyperparameters.items() if key not in {"cell_line_views", "drug_views"}
    }
    return ModelConfig(
        cell_line_featurizer=cell_line_featurizer,
        drug_featurizer=drug_featurizer,
        predictor=PredictorConfig(name=predictor_type, hyperparameters=predictor_hp),
    )


def naive_model_config(predictor_type: str) -> ModelConfig:
    """Map naive baseline names to modular configs."""
    return ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name=predictor_type),
    )


def legacy_model_config(model_name: str, hyperparameters: dict[str, Any]) -> ModelConfig:
    """Map literature model names to featurizer-aware zoo configs when available."""
    from drevalpy.models.zoo import list_zoo_names, zoo_model_config

    if model_name in list_zoo_names(include_external=True):
        return zoo_model_config(model_name, hyperparameters)
    predictor_type = LEGACY_PREDICTOR_BY_MODEL_NAME.get(model_name)
    if predictor_type is None:
        msg = f"Unknown legacy literature model name: {model_name}"
        raise KeyError(msg)
    return ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name=predictor_type, hyperparameters=hyperparameters),
    )


SKLEARN_PREDICTOR_BY_MODEL_NAME = {
    "ElasticNet": "elasticNet",
    "Lasso": "lasso",
    "RandomForest": "randomForest",
    "SVR": "svr",
    "GradientBoosting": "gradientBoosting",
    "AdaBoostDecisionTree": "adaboost",
    "KNNRegressor": "knn",
    "MultiViewRandomForest": "randomForest",
    "MultiViewXGBoost": "xgboost",
    "SingleDrugElasticNet": "elasticNet",
    "SingleDrugRandomForest": "randomForest",
}

NAIVE_PREDICTOR_BY_MODEL_NAME = {
    "NaivePredictor": "naiveMean",
    "NaiveDrugMeanPredictor": "naiveDrugMean",
    "NaiveCellLineMeanPredictor": "naiveCellLineMean",
    "NaiveTissueMeanPredictor": "naiveTissueMean",
    "NaiveTissueDrugMeanPredictor": "naiveTissueDrugMean",
    "NaiveMeanEffectsPredictor": "naiveMeanEffects",
}

LEGACY_PREDICTOR_BY_MODEL_NAME = {
    "DIPK": "dipk",
    "DrugGNN": "drugGNN",
    "MOLIR": "molir",
    "SuperFELTR": "superfeltr",
    "PharmaFormer": "pharmaFormer",
    "Precily": "precily",
    "SRMF": "srmf",
    "SimpleNeuralNetwork": "neuralNetwork",
    "MultiViewNeuralNetwork": "neuralNetwork",
}


def model_config_for_name(model_name: str, hyperparameters: dict[str, Any] | None = None) -> ModelConfig:
    """Resolve a legacy model name to a modular config, preferring zoo entries."""
    from drevalpy.models.zoo import list_zoo_names, zoo_model_config

    hp = hyperparameters or {}
    if model_name in list_zoo_names(include_external=True):
        return zoo_model_config(model_name, hp)
    if model_name in SKLEARN_PREDICTOR_BY_MODEL_NAME:
        return sklearn_model_config(SKLEARN_PREDICTOR_BY_MODEL_NAME[model_name], hp)
    if model_name in NAIVE_PREDICTOR_BY_MODEL_NAME:
        return naive_model_config(NAIVE_PREDICTOR_BY_MODEL_NAME[model_name])
    if model_name in LEGACY_PREDICTOR_BY_MODEL_NAME:
        return legacy_model_config(model_name, hp)
    msg = f"Unknown model name: {model_name}"
    raise KeyError(msg)


def sklearn_model_config_from_zoo(model_name: str, hyperparameters: dict[str, Any]) -> ModelConfig:
    """Compatibility helper that resolves sklearn baselines through the zoo when available."""
    from drevalpy.models.zoo import list_zoo_names, zoo_model_config

    if model_name in list_zoo_names(include_external=True):
        return zoo_model_config(model_name, hyperparameters)
    predictor_type = SKLEARN_PREDICTOR_BY_MODEL_NAME.get(model_name)
    if predictor_type is None:
        msg = f"Not a sklearn baseline model name: {model_name}"
        raise KeyError(msg)
    return sklearn_model_config(predictor_type, hyperparameters)
