"""Convert legacy checkpoint directories to component-stack format."""

from __future__ import annotations

from typing import Any

from drevalpy.models._component_bridge import ComponentDRPBridge, save_component_stack
from drevalpy.models._legacy_checkpoint_loaders import load_hyperparameters_json


def migrate_checkpoint_to_component_stack(
    model: Any,
    directory: str,
    *,
    output_directory: str | None = None,
) -> str:
    """Write a trained model's component stack to disk, optionally in a new directory."""
    target = output_directory or directory
    bridge: ComponentDRPBridge = getattr(model, "_bridge", None) or model._component_bridge
    if not bridge.is_trained():
        msg = "Cannot migrate checkpoint: component stack is not trained"
        raise RuntimeError(msg)
    hyperparameters = getattr(model, "hyperparameters", {}) or load_hyperparameters_json(directory)
    save_component_stack(bridge, target, hyperparameters=hyperparameters)
    return target
