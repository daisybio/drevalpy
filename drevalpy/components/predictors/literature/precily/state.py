"""Persistence mapping for Precily."""

from __future__ import annotations

from typing import Any

from drevalpy.components.predictors.literature._torch_state import load_state_dict, save_state_dict
from drevalpy.components.predictors.literature.precily.algorithm import PrecilyModel
from drevalpy.components.predictors.literature.precily.model_utils import PrecilyNetwork


def export_state(algorithm: PrecilyModel) -> dict[str, Any]:
    """Serialize a fitted Precily algorithm for predictor persistence."""
    payload: dict[str, Any] = {"hyperparameters": dict(algorithm.hyperparameters)}
    if algorithm.model is not None:
        first_layer = algorithm.model.net[0]
        payload["input_dim"] = int(first_layer.in_features)  # type: ignore[arg-type,misc]
        payload["model_state"] = save_state_dict(algorithm.model.state_dict())
    return payload


def apply_state(payload: dict[str, Any]) -> PrecilyModel:
    """Restore a Precily algorithm from a persisted payload."""
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        msg = "missing algorithm hyperparameters"
        raise ValueError(msg)
    algorithm = PrecilyModel()
    algorithm.configure(hyperparameters)
    input_dim = payload.get("input_dim")
    model_state = payload.get("model_state")
    if isinstance(input_dim, int) and isinstance(model_state, (bytes, bytearray)):
        algorithm.model = PrecilyNetwork(
            input_dim=input_dim,
            dropout=float(hyperparameters.get("dropout", 0.1)),
        )
        algorithm.model.load_state_dict(load_state_dict(bytes(model_state)))
    return algorithm
