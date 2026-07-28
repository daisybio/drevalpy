"""Persistence mapping for SuperFELTR."""

from __future__ import annotations

from typing import Any

from drevalpy.components.predictors.literature._torch_state import load_state_dict, save_state_dict
from drevalpy.components.predictors.literature.superfeltr.algorithm import SuperFELTR


def _save_module_state(algorithm: SuperFELTR, payload: dict[str, Any], attr: str) -> None:
    module = getattr(algorithm, attr, None)
    if module is not None and hasattr(module, "state_dict"):
        payload[f"{attr}_state"] = save_state_dict(module.state_dict())


def export_state(algorithm: SuperFELTR) -> dict[str, Any]:
    """Serialize a fitted SuperFELTR algorithm for predictor persistence."""
    payload: dict[str, Any] = {
        "hyperparameters": dict(algorithm.hyperparameters),
        "ranges": algorithm.ranges,
    }
    for attr in ("expr_encoder", "mut_encoder", "cnv_encoder", "regressor"):
        _save_module_state(algorithm, payload, attr)
    return payload


def apply_state(payload: dict[str, Any]) -> SuperFELTR:
    """Restore a SuperFELTR algorithm from a persisted payload."""
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        msg = "missing algorithm hyperparameters"
        raise ValueError(msg)
    algorithm = SuperFELTR()
    algorithm.configure(hyperparameters)
    ranges = payload.get("ranges")
    if isinstance(ranges, tuple):
        algorithm.ranges = ranges
    for attr in ("expr_encoder", "mut_encoder", "cnv_encoder", "regressor"):
        module = getattr(algorithm, attr, None)
        state_blob = payload.get(f"{attr}_state")
        if module is not None and isinstance(state_blob, (bytes, bytearray)):
            module.load_state_dict(load_state_dict(bytes(state_blob)))
    return algorithm
