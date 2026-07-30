"""Persistence mapping for PharmaFormer."""

from __future__ import annotations

from typing import Any

from drevalpy.components.predictors.literature._torch_state import load_state_dict, save_state_dict
from drevalpy.components.predictors.literature.pharmaformer.algorithm import PharmaFormerModel
from drevalpy.components.predictors.literature.pharmaformer.pharmaformer_training import _build_combined_model


def export_state(algorithm: PharmaFormerModel) -> dict[str, Any]:
    """Serialize a fitted PharmaFormer algorithm for predictor persistence."""
    payload: dict[str, Any] = {"hyperparameters": dict(algorithm.hyperparameters)}
    gene_input_size = getattr(algorithm, "_saved_gene_input_size", None)
    if gene_input_size is not None:
        payload["gene_input_size"] = int(gene_input_size)
    model = getattr(algorithm, "model", None)
    if model is not None and hasattr(model, "state_dict"):
        payload["model_state"] = save_state_dict(model.state_dict())
    return payload


def apply_state(payload: dict[str, Any]) -> PharmaFormerModel:
    """Restore a PharmaFormer algorithm from a persisted payload."""
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        msg = "missing algorithm hyperparameters"
        raise ValueError(msg)
    algorithm = PharmaFormerModel()
    algorithm.configure(hyperparameters)
    gene_input_size = payload.get("gene_input_size")
    model_state = payload.get("model_state")
    if isinstance(gene_input_size, int) and isinstance(model_state, (bytes, bytearray)):
        algorithm._saved_gene_input_size = gene_input_size
        algorithm.model = _build_combined_model(gene_input_size, hyperparameters, algorithm.DEVICE)
        algorithm.model.load_state_dict(load_state_dict(bytes(model_state)))
    return algorithm
