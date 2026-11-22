"""Load frozen v2 literature baseline outputs for regression tests.

Every file was recorded with random seed 0 (``_helpers.REFERENCE_SEED``).
"""

from __future__ import annotations

import json
from typing import Any

from upath import UPath

ROOT = UPath(__file__).parent


def load_reference() -> dict[str, Any]:
    """Load the frozen single-forward-pass baseline payload.

    :returns: The payload recorded in ``seed-0.json``.
    """
    return json.loads((ROOT / "seed-0.json").read_text(encoding="utf-8"))


def load_training() -> dict[str, Any]:
    """Load the frozen predictor-level tiny-fit evidence.

    :returns: Per-model predictions and checkpoint scores recorded in ``training.json``.
    """
    return json.loads((ROOT / "training.json").read_text(encoding="utf-8"))


def load_correction(correction: str) -> dict[str, Any]:
    """Load the tiny-fit evidence of a separately classified defect correction.

    :param correction: Key of the correction in ``training-corrections.json``.
    :returns: The correction's record, with its ``predictions`` and ``checkpoint_score``.
    """
    payload = json.loads((ROOT / "training-corrections.json").read_text(encoding="utf-8"))
    return payload[correction]
