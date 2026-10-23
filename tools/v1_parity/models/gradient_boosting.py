"""GradientBoosting parity case: 1.6.1 ``GradientBoosting`` against the v2 ``GradientBoosting`` zoo preset.

1.6.1 is ``HistGradientBoostingRegressor`` with ``validation_fraction=0.1``, ``n_iter_no_change=10`` and
``random_state=42``. sklearn switches early stopping on only above 10000 training rows (``early_stopping="auto"``),
and then draws the validation split from ``random_state``; v2 passes neither ``random_state`` nor the early-stopping
settings. The case therefore has two sizes. Both sides set ``max_iter=50``, ``learning_rate=0.1``, ``max_depth=4``:

``small``
    The first 3000 training pairs: no early stopping, no randomness. Parity.
``large_seed42``
    All training pairs (more than 10000): early stopping is active. The global numpy seed is 42 before
    v2 trains, which equals ``random_state=42`` on the 1.6.1 side. Parity.
``large``
    The same, v2 under the case seed instead of 42: the validation split differs. Deviation.
``defaults``
    The 1.6.1 first grid point (``max_iter=100``, ``max_depth=5``) against the unmodified v2 preset
    (``max_depth=6``) on the large set. Deviation.
"""

from __future__ import annotations

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import DEVIATION, StepSpec
from tools.v1_parity.models._sklearn_common import (
    build_sklearn_case,
    run_sklearn_v1,
    run_sklearn_v2,
    sklearn_steps,
)

NAME = "GradientBoosting"
V1_MODULE, V1_CLASS, V2_PRESET = "drevalpy.models.baselines.sklearn_models", "GradientBoosting", "GradientBoosting"

VARIANTS: dict[str, dict] = {
    "small": {
        "v1": {"max_iter": 50, "learning_rate": 0.1, "max_depth": 4},
        "v2": {"max_iter": 50, "learning_rate": 0.1, "max_depth": 4},
        "train_limit": 3000,
    },
    "large_seed42": {
        "v1": {"max_iter": 50, "learning_rate": 0.1, "max_depth": 4},
        "v2": {"max_iter": 50, "learning_rate": 0.1, "max_depth": 4},
        "np_seed": 42,
    },
    "large": {
        "v1": {"max_iter": 50, "learning_rate": 0.1, "max_depth": 4},
        "v2": {"max_iter": 50, "learning_rate": 0.1, "max_depth": 4},
        "v1_variant": "large_seed42",
        "scope": DEVIATION,
        "explanation": (
            "With more than 10000 training pairs sklearn early-stops on a validation split drawn from "
            "random_state. 1.6.1 pins random_state=42; v2 leaves it unset, so the split comes from the "
            "global numpy stream and differs from run to run unless the caller seeds it."
        ),
    },
    "defaults": {
        "v1": {"max_iter": 100, "learning_rate": 0.1, "max_depth": 5},
        "v2": {},
        "np_seed": 42,
        "scope": DEVIATION,
        "explanation": (
            "Hyperparameter defaults differ in max_depth: 1.6.1 first grid point is max_iter=100, "
            "learning_rate=0.1, max_depth=5 (grid 5/10/30, learning_rate 0.1/0.01); the v2 preset "
            "declares max_iter=100, learning_rate=0.1, max_depth=6."
        ),
    },
}

FINDINGS = [
    (
        "1.6.1 (#500) builds `HistGradientBoostingRegressor(max_iter, learning_rate, max_depth, "
        "validation_fraction=0.1, n_iter_no_change=10, random_state=42)`; the v2 `gradientBoosting` predictor "
        "passes max_depth, learning_rate and max_iter only. For fewer than 10000 training pairs sklearn does "
        "not early-stop and the two are identical (`small`). With more pairs early stopping is on and v2 draws "
        "its validation split from the global numpy stream instead of random_state=42: B5 must add "
        "`random_state=42`, `validation_fraction=0.1` and `n_iter_no_change=10` (the last two equal sklearn's "
        "defaults)."
    ),
    (
        "Default hyperparameters: 1.6.1 first grid point max_iter=100, learning_rate=0.1, max_depth=5 (grid "
        "max_depth 5/10/30); v2 declares max_iter=100, learning_rate=0.1, max_depth=6 (space 3..12). Only "
        "max_depth differs."
    ),
    (
        "1.6.1 selects the `landmark_genes` list while loading the CSVs (hyperparameter `gene_list`, a missing "
        "gene raises); the v2 preset consumes the whole gene expression view (`scaledGeneExpression` without "
        "`gene_list`). The case compares a pre-selected matrix; wiring the list into the preset belongs to B5."
    ),
    (
        "Features agree to float32 precision (class forward): v2 holds them as float32, 1.6.1 as float64. "
        "HistGradientBoosting bins the features itself in float64, and the bin boundaries agree."
    ),
]


def build_case(seed: int) -> Case:
    """Draw the shared inputs and record the variants.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    return build_sklearn_case(NAME, seed, VARIANTS, n_samples=1000, n_drugs=24, n_genes=30, n_bits=16)


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Parity steps for the aligned variants, deviation steps for the others.
    """
    return sklearn_steps(
        VARIANTS,
        attributes={"n_iter_": "exact", "train_score_": "exact", "validation_score_": "exact"},
        predictions="exact",
        random_state_note=(
            "1.6.1 pins random_state=42; v2 leaves it unset (None), which draws from the global numpy "
            "stream. The large_seed42 variant seeds that stream with 42 to reproduce 1.6.1; B5 should "
            "pin random_state=42 in the predictor."
        ),
    )


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the 1.6.1 model for every variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    return run_sklearn_v1(case, V1_MODULE, V1_CLASS)


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the v2 zoo preset for every variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    return run_sklearn_v2(case, V2_PRESET)
