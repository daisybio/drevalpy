"""Persistence mapping for DIPK."""

from __future__ import annotations

from typing import Any

from drevalpy.components.predictors.literature._torch_state import load_state_dict, save_state_dict
from drevalpy.components.predictors.literature.dipk.algorithm import DIPKModel


def export_state(algorithm: DIPKModel) -> dict[str, Any]:
    """Serialize a fitted algorithm for predictor persistence.

    :param algorithm: Fitted algorithm instance.

    :returns: JSON-serializable state mapping.
    """
    payload: dict[str, Any] = {"hyperparameters": dict(algorithm.hyperparameters)}
    model = getattr(algorithm, "model", None)
    if model is not None and hasattr(model, "state_dict"):
        payload["model_state"] = save_state_dict(model.state_dict())
    return payload


def apply_state(payload: dict[str, Any]) -> DIPKModel:
    """Restore an algorithm from a persisted payload.

    :param payload: Serialized state produced by ``export_state``.

    :returns: Configured algorithm instance.

    :raises ValueError: If required payload fields are missing.
    """
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        msg = "missing algorithm hyperparameters"
        raise ValueError(msg)
    algorithm = DIPKModel()
    algorithm.configure(hyperparameters)
    model_state = payload.get("model_state")
    model = getattr(algorithm, "model", None)
    if isinstance(model_state, (bytes, bytearray)) and model is not None:
        model.load_state_dict(load_state_dict(bytes(model_state)))
    return algorithm
