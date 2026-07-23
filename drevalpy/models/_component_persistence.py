"""Versioned persistence for a configured component stack.

Joblib checkpoints are trusted-input-only: callers must only load artifacts they
created with ``save_composed_model`` in the same drevalpy version family.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib

from drevalpy.models.config import ModelConfig

if TYPE_CHECKING:
    from drevalpy.models.composed_model import ComposedModel

FORMAT_NAME = "drevalpy-composed-model"
FORMAT_VERSION = 1
STATE_FILE = "composed_model.joblib"


class ComposedModelCheckpointError(Exception):
    """Base error for native composed-model checkpoint problems."""


class UnsupportedCheckpointFormatError(ComposedModelCheckpointError, ValueError):
    """Raised when checkpoint format or version is not supported."""


class CorruptedCheckpointError(ComposedModelCheckpointError, ValueError):
    """Raised when checkpoint payload structure or content is invalid."""


def save_composed_model(model: ComposedModel, directory: str) -> None:
    """Save config and component state in one self-describing payload."""
    if model.config is None:
        raise RuntimeError("Cannot save a composed model without its ModelConfig")
    if not model.is_fitted():
        raise RuntimeError("Cannot save: component stack is not trained")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "config": model.config.model_dump(mode="json"),
        "state": model.component_state(),
    }
    joblib.dump(payload, target / STATE_FILE)


def load_composed_model(directory: str) -> ComposedModel:
    """Load only the canonical component-stack format."""
    path = Path(directory) / STATE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing native composed-model checkpoint: {path}")
    try:
        payload: Any = joblib.load(path)
    except Exception as exc:
        raise CorruptedCheckpointError(f"Failed to deserialize checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CorruptedCheckpointError("checkpoint payload is not a mapping")
    if payload.get("format") != FORMAT_NAME or payload.get("version") != FORMAT_VERSION:
        raise UnsupportedCheckpointFormatError(
            f"unsupported checkpoint format/version: {payload.get('format')!r}/{payload.get('version')!r}"
        )
    try:
        config = ModelConfig.model_validate(payload["config"])
    except Exception as exc:
        raise CorruptedCheckpointError("checkpoint config is invalid") from exc
    state = payload.get("state")
    if not isinstance(state, dict):
        raise CorruptedCheckpointError("checkpoint state is not a mapping")
    model = config.create_model()
    try:
        model.restore_component_state(state)
    except (ValueError, RuntimeError) as exc:
        raise CorruptedCheckpointError(
            f"checkpoint component state is invalid: {exc}" if str(exc) else "checkpoint component state is invalid"
        ) from exc
    if not model.is_fitted():
        raise CorruptedCheckpointError("checkpoint did not restore a fitted predictor")
    return model
