"""Build :class:`~drevalpy.components.config.ModelConfig` objects for existing models."""

from __future__ import annotations

from typing import Any

from drevalpy.components.config import FeaturizerConfig, ModelConfig, PredictorConfig


def _get_view_as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _views(hyperparameters: dict[str, Any], key: str, default: list[str]) -> list[str]:
    return _get_view_as_list(hyperparameters.get(key, default))


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


def legacy_model_config(predictor_type: str, hyperparameters: dict[str, Any]) -> ModelConfig:
    """Map literature/neural monolithic models to legacy stack predictors."""
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
    "SimpleNeuralNetwork": "simpleNeuralNetwork",
    "MultiViewNeuralNetwork": "multiViewNeuralNetwork",
}
