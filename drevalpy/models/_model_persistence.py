"""Versioned persistence for concrete DRPModel instances.

Joblib checkpoints are trusted-input-only: callers must only load artifacts they
created with ``save_model`` in the same drevalpy version family.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib

from drevalpy.models.config import ModelConfig

if TYPE_CHECKING:
    from drevalpy.models.drp_model import DRPModel

FORMAT_NAME = "drevalpy-model"
FORMAT_VERSION = 1
STATE_FILE = "model.joblib"


class ModelCheckpointError(Exception):
    """Base error for DRPModel checkpoint problems."""


class UnsupportedCheckpointFormatError(ModelCheckpointError, ValueError):
    """Raised when checkpoint format or version is not supported."""


class CorruptedCheckpointError(ModelCheckpointError, ValueError):
    """Raised when checkpoint payload structure or content is invalid."""


class IncompatibleModelCheckpointError(ModelCheckpointError, ValueError):
    """Raised when checkpoint model identity does not match the loader class."""


def save_model(model: DRPModel, directory: str) -> None:
    """Save model identity, config, and component state in one payload."""
    stack = model._stack
    if stack is None or not stack.is_fitted():
        raise RuntimeError("Cannot save: component stack is not trained")
    config = model._resolved_model_config
    if config is None:
        raise RuntimeError("Cannot save a model without its ModelConfig")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "model_name": model.get_model_name(),
        "config": config.model_dump(mode="json"),
        "state": stack.component_state(),
    }
    joblib.dump(payload, target / STATE_FILE)


def load_model_payload(directory: str) -> tuple[str, ModelConfig, dict[str, object]]:
    """Load and validate a DRPModel checkpoint payload."""
    path = Path(directory) / STATE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {path}")
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
    model_name = payload.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise CorruptedCheckpointError("checkpoint model_name is missing or invalid")
    try:
        config = ModelConfig.model_validate(payload["config"])
    except Exception as exc:
        raise CorruptedCheckpointError("checkpoint config is invalid") from exc
    state = payload.get("state")
    if not isinstance(state, dict):
        raise CorruptedCheckpointError("checkpoint state is not a mapping")
    return model_name, config, state
