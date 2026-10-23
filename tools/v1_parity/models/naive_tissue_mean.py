"""Parity case: 1.6.1 ``NaiveTissueMeanPredictor`` against the v2 ``NaiveTissueMeanPredictor`` preset.

The model predicts the mean per tissue, the dataset mean for an unseen tissue.
It is compared end to end on the training and the test pairs (see ``_naive_common`` for the
fallbacks the case exercises).
"""

from __future__ import annotations

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import StepSpec
from tools.v1_parity.models._naive_common import (
    COMMON_FINDING,
    build_naive_case,
    naive_steps,
    run_naive_v1,
    run_naive_v2,
)

NAME = "NaiveTissueMeanPredictor"
V1_CLASS = "NaiveTissueMeanPredictor"
V2_PRESET = "NaiveTissueMeanPredictor"
TOLERANCE = "exact"

VARIANTS: dict[str, dict[str, bool]] = {"default": {"tissue": True}}

FINDINGS: list[str] = [
    COMMON_FINDING,
]


def build_case(seed: int) -> Case:
    """Draw the shared inputs, with every fallback of the model in the test pairs.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    return build_naive_case(NAME, seed, VARIANTS)


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Parity steps for the dataset mean and the predictions.
    """
    return naive_steps(VARIANTS, TOLERANCE)


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the 1.6.1 model.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    return run_naive_v1(case, V1_CLASS)


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the v2 zoo preset.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    return run_naive_v2(case, V2_PRESET)
