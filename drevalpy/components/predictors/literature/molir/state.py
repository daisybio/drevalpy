"""Persistence mapping for the molir algorithm."""

from __future__ import annotations

from typing import Any

from drevalpy.components.predictors.literature._torch_state import load_state_dict, save_state_dict
from drevalpy.components.predictors.literature.molir.algorithm import MOLIR


def export_state(algorithm: MOLIR) -> dict[str, Any]:
    """Serialize a fitted MOLIR algorithm for predictor persistence."""
    payload: dict[str, Any] = {
        "hyperparameters": dict(algorithm.hyperparameters),
        "preload": {
            name: getattr(algorithm, name)
            for name in (
                "layer_connections",
                "gene2id_mapping_ont",
                "ontology_gene_order",
                "gene_dim_input",
            )
            if getattr(algorithm, name, None) is not None
        },
    }
    model = getattr(algorithm, "model", None)
    if model is not None and hasattr(model, "state_dict"):
        payload["model_state"] = save_state_dict(model.state_dict())
    encoder = getattr(algorithm, "gene_expression_encoder", None)
    if encoder is not None and hasattr(encoder, "state_dict"):
        payload["gene_expression_encoder_state"] = save_state_dict(encoder.state_dict())
    return payload


def apply_state(payload: dict[str, Any]) -> MOLIR:
    """Restore a MOLIR algorithm from a persisted payload."""
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        msg = "missing algorithm hyperparameters"
        raise ValueError(msg)
    algorithm = MOLIR()
    preload = payload.get("preload")
    if isinstance(preload, dict):
        for key, value in preload.items():
            setattr(algorithm, key, value)
    algorithm.configure(hyperparameters)
    model_state = payload.get("model_state")
    model = getattr(algorithm, "model", None)
    if isinstance(model_state, (bytes, bytearray)) and model is not None:
        model.load_state_dict(load_state_dict(bytes(model_state)))
    encoder_state = payload.get("gene_expression_encoder_state")
    encoder = getattr(algorithm, "gene_expression_encoder", None)
    if isinstance(encoder_state, (bytes, bytearray)) and encoder is not None:
        encoder.load_state_dict(load_state_dict(bytes(encoder_state)))
    return algorithm
