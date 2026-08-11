"""Serialize torch state dicts for predictor persistence."""

from __future__ import annotations

from drevalpy.utils.torch_io import (
    load_state_dict,
    load_trusted_mapping,
    save_state_dict,
    save_trusted_mapping,
)

__all__ = [
    "load_state_dict",
    "load_trusted_mapping",
    "save_state_dict",
    "save_trusted_mapping",
]
