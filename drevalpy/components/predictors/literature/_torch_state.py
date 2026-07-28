"""Serialize torch state dicts for predictor persistence."""

from __future__ import annotations

import io
from typing import Any

import torch


def save_state_dict(state_dict: dict[str, Any]) -> bytes:
    """Serialize a PyTorch state dict to bytes."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def load_state_dict(blob: bytes) -> dict[str, Any]:
    """Load a PyTorch state dict from bytes."""
    data = torch.load(io.BytesIO(blob), weights_only=True)  # noqa: S614
    if not isinstance(data, dict):
        msg = "torch payload must be a state dict mapping"
        raise TypeError(msg)
    return data


def save_object_mapping(payload: dict[str, Any]) -> bytes:
    """Serialize an arbitrary mapping with torch.save."""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def load_object_mapping(blob: bytes) -> dict[str, Any]:
    """Load a mapping previously written with save_object_mapping."""
    data = torch.load(io.BytesIO(blob), weights_only=False)  # noqa: S614
    if not isinstance(data, dict):
        msg = "torch payload must be a mapping"
        raise TypeError(msg)
    return data
