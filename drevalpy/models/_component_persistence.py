"""Versioned persistence for a configured component stack."""

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
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload is not a mapping")
        if payload.get("format") != FORMAT_NAME or payload.get("version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported checkpoint format/version: {payload.get('format')!r}/{payload.get('version')!r}"
            )
        config = ModelConfig.model_validate(payload["config"])
        state = payload["state"]
        if not isinstance(state, dict):
            raise ValueError("checkpoint state is not a mapping")
        model = config.create_model()
        model.restore_component_state(state)
        if not model.is_fitted():
            raise ValueError("checkpoint did not restore a fitted predictor")
        return model
    except (FileNotFoundError, ImportError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to load native composed-model checkpoint {path}: {exc}") from exc
