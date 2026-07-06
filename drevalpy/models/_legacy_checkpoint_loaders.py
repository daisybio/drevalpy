"""Load legacy and native checkpoint artifacts into component stacks."""

from __future__ import annotations

import json
import os
from typing import Any

import joblib

from drevalpy.models._component_bridge import (
    _COMPONENT_STACK_FILE,
    _HYPERPARAMETERS_FILE,
    ComponentDRPBridge,
    load_component_stack,
    restore_naive_to_components,
    restore_sklearn_to_components,
)
from drevalpy.models.factory import NAIVE_PREDICTOR_BY_MODEL_NAME


def has_component_stack(directory: str) -> bool:
    return os.path.exists(os.path.join(directory, _COMPONENT_STACK_FILE))


def load_hyperparameters_json(directory: str) -> dict[str, Any]:
    path = os.path.join(directory, _HYPERPARAMETERS_FILE)
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def load_native_checkpoint(model: Any, directory: str) -> dict[str, Any]:
    """Load a component-stack checkpoint into *model*'s bridge."""
    bridge: ComponentDRPBridge = getattr(model, "_bridge", None) or getattr(model, "_component_bridge")
    hyperparameters = load_hyperparameters_json(directory)
    if hyperparameters:
        model.build_model(hyperparameters)
    elif bridge.composed is None:
        model.build_model({})
    loaded_hp = load_component_stack(bridge, directory)
    return loaded_hp or hyperparameters


def naive_predictor_type_for_model(model: Any) -> str:
    predictor_type = NAIVE_PREDICTOR_BY_MODEL_NAME.get(model.get_model_name())
    if predictor_type is None:
        msg = f"No component predictor registered for {model.get_model_name()!r}"
        raise ValueError(msg)
    return predictor_type


def load_legacy_naive_checkpoint(model: Any, directory: str) -> None:
    path = os.path.join(directory, "naive_model.json")
    with open(path) as handle:
        config = json.load(handle)
    model.build_model({})
    for attr in (
        "dataset_mean",
        "drug_means",
        "cell_line_means",
        "tissue_means",
        "tissue_drug_means",
        "cell_line_effects",
        "drug_effects",
        "tissue_effects",
    ):
        if attr in config:
            value = config[attr]
            if attr == "tissue_drug_means":
                value = {tuple(key.split("|", maxsplit=1)): mean for key, mean in value.items()}
            object.__setattr__(model, f"_legacy_{attr}", value)
    restore_naive_to_components(model, naive_predictor_type_for_model(model))


def load_legacy_sklearn_checkpoint(model: Any, directory: str) -> None:
    model_path = os.path.join(directory, "model.pkl")
    if not os.path.exists(model_path):
        msg = f"{model_path} not found"
        raise FileNotFoundError(msg)
    hyperparameters = load_hyperparameters_json(directory)
    model.build_model(hyperparameters)
    legacy_state: dict[str, Any] = {"model": joblib.load(model_path)}
    scaler_path = os.path.join(directory, "scaler.pkl")
    if os.path.exists(scaler_path):
        legacy_state["gene_expression_scaler"] = joblib.load(scaler_path)
    methylation_scaler_path = os.path.join(directory, "methylation_scaler.pkl")
    if os.path.exists(methylation_scaler_path):
        legacy_state["methylation_scaler"] = joblib.load(methylation_scaler_path)
    methylation_pca_path = os.path.join(directory, "methylation_pca.pkl")
    if os.path.exists(methylation_pca_path):
        legacy_state["methylation_pca"] = joblib.load(methylation_pca_path)
    transformer_path = os.path.join(directory, "proteomics_transformer.pkl")
    if os.path.exists(transformer_path):
        legacy_state["proteomics_transformer"] = joblib.load(transformer_path)
    model._legacy_sklearn_state = legacy_state
    restore_sklearn_to_components(model)
