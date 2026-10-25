"""Load frozen v2 literature baseline outputs for regression tests."""

from __future__ import annotations

import json
from typing import Any

from upath import UPath

ROOT = UPath(__file__).parent


def load_seed(seed: int) -> dict[str, Any]:
    """Load one deterministic baseline payload."""
    if seed not in (0, 1, 2):
        raise ValueError("baseline seeds are 0, 1, and 2")
    return json.loads((ROOT / f"seed-{seed}.json").read_text(encoding="utf-8"))


def load_training_seed(seed: int) -> dict[str, Any]:
    """Load one seed's frozen predictor-level tiny-fit evidence."""
    if seed not in (0, 1, 2):
        raise ValueError("baseline seeds are 0, 1, and 2")
    payload = json.loads((ROOT / "training.json").read_text(encoding="utf-8"))
    return payload["seeds"][str(seed)]


def load_corrected_training_seed(correction: str, seed: int) -> dict[str, Any]:
    """Load one seed for a separately classified defect correction."""
    if seed not in (0, 1, 2):
        raise ValueError("correction seeds are 0, 1, and 2")
    payload = json.loads((ROOT / "training-corrections.json").read_text(encoding="utf-8"))
    return payload[correction]["seeds"][str(seed)]
