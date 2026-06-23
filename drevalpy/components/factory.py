"""Build :class:`~drevalpy.components.config.ModelConfig` objects for existing models."""

from __future__ import annotations

from typing import Any

from drevalpy.components.config import FeaturizerConfig, ModelConfig, PredictorConfig


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
        if len(cell_line_views) == 1 and cell_line_views[0] == "gene_expression":
            cell_line_type = "scaledGeneExpression"
            cell_line_hp: dict[str, Any] = {"view": "gene_expression"}
        elif len(cell_line_views) == 1 and cell_line_views[0] == "proteomics":
            cell_line_type = "proteomics"
            cell_line_hp = {"view": "proteomics"}
            if "proteomics_feature_threshold" in hyperparameters:
                cell_line_hp["proteomics_feature_threshold"] = hyperparameters["proteomics_feature_threshold"]
            if "proteomics_n_features" in hyperparameters:
                cell_line_hp["proteomics_n_features"] = hyperparameters["proteomics_n_features"]
            if "proteomics_normalization_width" in hyperparameters:
                cell_line_hp["proteomics_normalization_width"] = hyperparameters[
                    "proteomics_normalization_width"
                ]
            if "proteomics_normalization_downshift" in hyperparameters:
                cell_line_hp["proteomics_normalization_downshift"] = hyperparameters[
                    "proteomics_normalization_downshift"
                ]
        elif len(cell_line_views) == 1:
            cell_line_type = "view"
            cell_line_hp = {"view": cell_line_views[0]}
        else:
            cell_line_type = "multiConcat"
            cell_line_hp = {"views": cell_line_views}
            if "methylation_n_components" in hyperparameters:
                cell_line_hp["methylation_n_components"] = hyperparameters["methylation_n_components"]
            if "methylation_pca_components" in hyperparameters:
                cell_line_hp["methylation_n_components"] = hyperparameters["methylation_pca_components"]
            if "proteomics_feature_threshold" in hyperparameters:
                cell_line_hp["proteomics_feature_threshold"] = hyperparameters["proteomics_feature_threshold"]
            if "proteomics_n_features" in hyperparameters:
                cell_line_hp["proteomics_n_features"] = hyperparameters["proteomics_n_features"]
            if "proteomics_normalization_width" in hyperparameters:
                cell_line_hp["proteomics_normalization_width"] = hyperparameters[
                    "proteomics_normalization_width"
                ]
            if "proteomics_normalization_downshift" in hyperparameters:
                cell_line_hp["proteomics_normalization_downshift"] = hyperparameters[
                    "proteomics_normalization_downshift"
                ]
        cell_line_featurizer = FeaturizerConfig(
            type=cell_line_type,
            registry="cell_line",
            hyperparameters=cell_line_hp,
        )

    if "drug_views" in hyperparameters:
        drug_views = _views(hyperparameters, "drug_views", ["fingerprints"])
        if drug_views:
            drug_type = "fingerprints" if drug_views[0] == "fingerprints" else "view"
            drug_featurizer = FeaturizerConfig(
                type=drug_type,
                registry="drug",
                view=drug_views[0],
            )

    return cell_line_featurizer, drug_featurizer


def sklearn_model_config(predictor_type: str, hyperparameters: dict[str, Any]) -> ModelConfig:
    """Map sklearn baseline hyperparameters to a modular config."""
    cell_line_views = _views(hyperparameters, "cell_line_views", ["gene_expression"])
    drug_views = _views(hyperparameters, "drug_views", ["fingerprints"])
    if len(cell_line_views) == 1 and cell_line_views[0] == "gene_expression":
        cell_line_type = "scaledGeneExpression"
        cell_line_hp: dict[str, Any] = {"view": "gene_expression"}
    elif len(cell_line_views) == 1 and cell_line_views[0] == "proteomics":
        cell_line_type = "proteomics"
        cell_line_hp = {"view": "proteomics"}
        if "proteomics_feature_threshold" in hyperparameters:
            cell_line_hp["proteomics_feature_threshold"] = hyperparameters["proteomics_feature_threshold"]
        if "proteomics_n_features" in hyperparameters:
            cell_line_hp["proteomics_n_features"] = hyperparameters["proteomics_n_features"]
        if "proteomics_normalization_width" in hyperparameters:
            cell_line_hp["proteomics_normalization_width"] = hyperparameters[
                "proteomics_normalization_width"
            ]
        if "proteomics_normalization_downshift" in hyperparameters:
            cell_line_hp["proteomics_normalization_downshift"] = hyperparameters[
                "proteomics_normalization_downshift"
            ]
    elif len(cell_line_views) == 1:
        cell_line_type = "view"
        cell_line_hp = {"view": cell_line_views[0]}
    else:
        cell_line_type = "multiConcat"
        cell_line_hp = {"views": cell_line_views}
        if "methylation_n_components" in hyperparameters:
            cell_line_hp["methylation_n_components"] = hyperparameters["methylation_n_components"]
        if "proteomics_feature_threshold" in hyperparameters:
            cell_line_hp["proteomics_feature_threshold"] = hyperparameters["proteomics_feature_threshold"]
        if "proteomics_n_features" in hyperparameters:
            cell_line_hp["proteomics_n_features"] = hyperparameters["proteomics_n_features"]
        if "proteomics_normalization_width" in hyperparameters:
            cell_line_hp["proteomics_normalization_width"] = hyperparameters[
                "proteomics_normalization_width"
            ]
        if "proteomics_normalization_downshift" in hyperparameters:
            cell_line_hp["proteomics_normalization_downshift"] = hyperparameters[
                "proteomics_normalization_downshift"
            ]

    drug_featurizer = None
    if drug_views:
        if drug_views[0] == "fingerprints":
            drug_type = "fingerprints"
        else:
            drug_type = "view"
        drug_featurizer = FeaturizerConfig(
            type=drug_type,
            registry="drug",
            view=drug_views[0],
        )

    predictor_hp = {
        key: value
        for key, value in hyperparameters.items()
        if key not in {"cell_line_views", "drug_views"}
    }
    return ModelConfig(
        cell_line_featurizer=FeaturizerConfig(type=cell_line_type, registry="cell_line", hyperparameters=cell_line_hp),
        drug_featurizer=drug_featurizer,
        predictor=PredictorConfig(type=predictor_type, hyperparameters=predictor_hp),
    )


def naive_model_config(predictor_type: str) -> ModelConfig:
    """Map naive baseline names to modular configs."""
    return ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(type=predictor_type),
    )


def legacy_model_config(model_name: str, hyperparameters: dict[str, Any]) -> ModelConfig:
    """Map literature model names to featurizer-aware zoo configs when available."""
    from drevalpy.components.zoo import list_zoo_names, zoo_model_config

    if model_name in list_zoo_names(include_external=True):
        return zoo_model_config(model_name, hyperparameters)
    predictor_type = LEGACY_PREDICTOR_BY_MODEL_NAME.get(model_name)
    if predictor_type is None:
        msg = f"Unknown legacy literature model name: {model_name}"
        raise KeyError(msg)
    return ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(type=predictor_type, hyperparameters=hyperparameters),
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
    from drevalpy.components.zoo import get_zoo_config, list_zoo_names, zoo_model_config

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
    from drevalpy.components.zoo import list_zoo_names, zoo_model_config

    if model_name in list_zoo_names(include_external=True):
        return zoo_model_config(model_name, hyperparameters)
    predictor_type = SKLEARN_PREDICTOR_BY_MODEL_NAME.get(model_name)
    if predictor_type is None:
        msg = f"Not a sklearn baseline model name: {model_name}"
        raise KeyError(msg)
    return sklearn_model_config(predictor_type, hyperparameters)
