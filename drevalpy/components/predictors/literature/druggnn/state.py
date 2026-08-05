"""Persistence mapping for DrugGNN."""

from __future__ import annotations

from typing import Any

from drevalpy.components.predictors.literature._torch_state import load_state_dict, save_state_dict
from drevalpy.components.predictors.literature.druggnn.algorithm import DrugGNN, DrugGNNModule


def export_state(algorithm: DrugGNN) -> dict[str, Any]:
    """Serialize a fitted algorithm for predictor persistence.

    :param algorithm: Fitted algorithm instance.

    :returns: JSON-serializable state mapping.
    """
    payload: dict[str, Any] = {"hyperparameters": dict(algorithm.hyperparameters)}
    if algorithm.model is not None:
        payload["model_state"] = save_state_dict(algorithm.model.state_dict())
        payload["architecture"] = {
            "num_node_features": int(getattr(algorithm, "_saved_num_node_features", 0)),
            "num_cell_features": int(getattr(algorithm, "_saved_num_cell_features", 0)),
            "hidden_dim": int(algorithm.hyperparameters.get("hidden_dim", 64)),
            "dropout": float(algorithm.hyperparameters.get("dropout", 0.2)),
            "learning_rate": float(algorithm.hyperparameters.get("learning_rate", 0.001)),
        }
    return payload


def apply_state(payload: dict[str, Any]) -> DrugGNN:
    """Restore an algorithm from a persisted payload.

    :param payload: Serialized state produced by ``export_state``.

    :returns: Configured algorithm instance.

    :raises ValueError: If required payload fields are missing.
    """
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        msg = "missing algorithm hyperparameters"
        raise ValueError(msg)
    algorithm = DrugGNN()
    algorithm.configure(hyperparameters)
    architecture = payload.get("architecture")
    model_state = payload.get("model_state")
    if isinstance(architecture, dict) and isinstance(model_state, (bytes, bytearray)):
        algorithm.model = DrugGNNModule(
            num_node_features=int(architecture["num_node_features"]),
            num_cell_features=int(architecture["num_cell_features"]),
            hidden_dim=int(architecture.get("hidden_dim", 64)),
            dropout=float(architecture.get("dropout", 0.2)),
            learning_rate=float(architecture.get("learning_rate", 0.001)),
        )
        algorithm._saved_num_node_features = int(architecture["num_node_features"])
        algorithm._saved_num_cell_features = int(architecture["num_cell_features"])
        algorithm.model.load_state_dict(load_state_dict(bytes(model_state)))
    return algorithm
