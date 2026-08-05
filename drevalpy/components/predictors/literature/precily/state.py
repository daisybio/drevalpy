"""Persistence mapping for Precily."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from drevalpy.components.predictors.literature._torch_state import load_state_dict, save_state_dict
from drevalpy.components.predictors.literature.precily.algorithm import PrecilyModel
from drevalpy.components.predictors.literature.precily.model_utils import PrecilyNetwork


def export_state(algorithm: PrecilyModel) -> dict[str, Any]:
    """Serialize a fitted algorithm for predictor persistence.

    :param algorithm: Fitted algorithm instance.

    :returns: JSON-serializable state mapping.

    :raises TypeError: If the saved Precily network has an unexpected first layer.
    """
    payload: dict[str, Any] = {"hyperparameters": dict(algorithm.hyperparameters)}
    if algorithm.model is not None:
        first_layer = algorithm.model.net[0]
        if not isinstance(first_layer, nn.Linear):
            msg = "PrecilyNetwork must start with a Linear layer"
            raise TypeError(msg)
        payload["input_dim"] = int(first_layer.in_features)
        payload["model_state"] = save_state_dict(algorithm.model.state_dict())
    return payload


def apply_state(payload: dict[str, Any]) -> PrecilyModel:
    """Restore an algorithm from a persisted payload.

    :param payload: Serialized state produced by ``export_state``.

    :returns: Configured algorithm instance.

    :raises ValueError: If required payload fields are missing.
    """
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
