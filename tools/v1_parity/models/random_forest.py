"""RandomForest parity case: 1.6.1 ``RandomForest`` against the v2 ``RandomForest`` zoo preset.

Variants (both sides set the hyperparameters explicitly):

``aligned``
    A small forest (30 trees, ``max_samples=0.5``, ``max_depth=6``, ``n_jobs=1``). 1.6.1 leaves
    ``random_state`` unset and so draws from the global numpy stream; v2 gets ``random_state`` equal to the
    seed that stream is started with, which yields the identical forests.
``unbounded``
    ``max_depth=None`` (1.6.1 spells it ``"None"`` in the grid and converts it).
``defaults``
    The 1.6.1 first grid point against the unmodified v2 preset. Recorded as a deviation.
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

NAME = "RandomForest"
V1_MODULE, V1_CLASS, V2_PRESET = "drevalpy.models.baselines.sklearn_models", "RandomForest", "RandomForest"

VARIANTS: dict[str, dict] = {
    "aligned": {
        "v1": {"n_estimators": 30, "criterion": "squared_error", "max_samples": 0.5, "max_depth": 6, "n_jobs": 1},
        "v2": {
            "n_estimators": 30,
            "max_samples": 0.5,
            "max_depth": 6,
            "criterion": "squared_error",
            "n_jobs": 1,
            "random_state": "$seed",
        },
    },
    "unbounded": {
        "v1": {"n_estimators": 30, "criterion": "squared_error", "max_samples": 0.5, "max_depth": "None", "n_jobs": 1},
        "v2": {
            "n_estimators": 30,
            "max_samples": 0.5,
            "max_depth": None,
            "criterion": "squared_error",
            "n_jobs": 1,
            "random_state": "$seed",
        },
    },
    "defaults": {
        "v1": {"n_estimators": 100, "criterion": "squared_error", "max_samples": 0.2, "max_depth": 5, "n_jobs": -1},
        "v2": {},
        "scope": DEVIATION,
        "explanation": (
            "Hyperparameter defaults differ: 1.6.1 first grid point is n_estimators=100, "
            "max_samples=0.2, max_depth=5 (grid 5/10/30), n_jobs=-1, random_state unset; the v2 preset "
            "has max_depth=15 and a pinned random_state (see the random_state step). The forests are "
            "different models."
        ),
    },
}

FINDINGS = [
    (
        "1.6.1 `RandomForest` passes n_estimators, criterion, max_samples, max_depth and n_jobs only; "
        "`random_state` is unset (global numpy stream). The v2 `randomForest` predictor pins `random_state` (an "
        "integer; `None` before the working-tree change that sets 42). With `random_state` equal to the integer "
        "the global numpy stream is seeded with, the forests are identical."
    ),
    (
        "Default hyperparameters: 1.6.1 first grid point n_estimators=100, max_samples=0.2, max_depth=5, "
        "criterion=squared_error, n_jobs=-1 (grid max_depth 5/10/30); v2 declares n_estimators=100, "
        "max_samples=0.2, max_depth=15 (space 5..25), criterion=squared_error, n_jobs=-1. Only max_depth "
        "differs. B5 should decide whether to match 5."
    ),
    (
        "1.6.1 selects the `landmark_genes` list while loading the CSVs (hyperparameter `gene_list`, a missing "
        "gene raises); the v2 preset consumes the whole gene expression view (`scaledGeneExpression` without "
        "`gene_list`). The case compares a pre-selected matrix; wiring the list into the preset belongs to B5."
    ),
    (
        "Features agree to float32 precision (class forward): v2 holds them as float32, 1.6.1 as float64. The "
        "forests nevertheless have the same structure, because sklearn casts the training matrix to float32 "
        "itself and the split order of random data does not change under a 1-ulp perturbation."
    ),
]


def build_case(seed: int) -> Case:
    """Draw the shared inputs and record the variants.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    return build_sklearn_case(NAME, seed, VARIANTS, n_samples=60, n_drugs=8, n_genes=50, n_bits=32)


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Parity steps for the aligned variants, deviation steps for the others.
    """
    return sklearn_steps(
        VARIANTS,
        attributes={"feature_importances_": "exact"},
        predictions="exact",
        random_state_note=(
            "1.6.1: unset (global numpy stream, seeded with the case seed). v2: an integer; the case "
            "passes the case seed, which starts the identical stream. The preset's own value is "
            "recorded in the `defaults` variant."
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
