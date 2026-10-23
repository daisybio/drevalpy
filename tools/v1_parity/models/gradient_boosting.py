"""GradientBoosting parity case: 1.6.1 ``GradientBoosting`` against the v2 ``GradientBoosting`` zoo preset.

1.6.1 is ``HistGradientBoostingRegressor`` with ``validation_fraction=0.1``, ``n_iter_no_change=10`` and
``random_state=42``. sklearn switches early stopping on only above 10000 training rows (``early_stopping="auto"``),
and then draws the validation split from ``random_state``. The v2 predictor passes the same three values
(``non_tunable_hyperparameters``), so the case has two sizes:

``small``
    The first 3000 training pairs: no early stopping, no randomness. ``max_iter=50``, ``learning_rate=0.1``,
    ``max_depth=4`` on both sides.
``large``
    All training pairs (more than 10000): early stopping is active and its validation split comes from
    ``random_state=42`` on both sides. Same hyperparameters as ``small``.
``defaults``
    The 1.6.1 first grid point (``max_iter=100``, ``learning_rate=0.1``, ``max_depth=5``) against the unmodified
    v2 preset on the large set.
"""

from __future__ import annotations

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import PARITY, StepSpec
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
    "large": {
        "v1": {"max_iter": 50, "learning_rate": 0.1, "max_depth": 4},
        "v2": {"max_iter": 50, "learning_rate": 0.1, "max_depth": 4},
    },
    "defaults": {"v1": {"max_iter": 100, "learning_rate": 0.1, "max_depth": 5}, "v2": {}},
}

FINDINGS = [
    (
        "1.6.1 (#500) builds `HistGradientBoostingRegressor(max_iter, learning_rate, max_depth, "
        "validation_fraction=0.1, n_iter_no_change=10, random_state=42)`; the v2 `gradientBoosting` predictor "
        "passes the same values (the last three are `non_tunable_hyperparameters`). For fewer than 10000 training "
        "pairs sklearn does not early-stop and the two are identical (`small`); with more pairs early stopping is "
        "on and both draw the validation split from `random_state=42` (`large`), so n_iter_, the train and "
        "validation scores and the predictions agree exactly."
    ),
    (
        "Default hyperparameters agree: 1.6.1 first grid point max_iter=100, learning_rate=0.1, max_depth=5 (grid "
        "max_depth 5/10/30); the v2 defaults are max_iter=100, learning_rate=0.1, max_depth=5 (space 3..30). The "
        "`defaults` variant runs the unmodified preset."
    ),
    (
        "1.6.1 selects the `landmark_genes` list while loading the CSVs (hyperparameter `gene_list`, a missing "
        "gene raises); the v2 preset selects the same list with `scaledGeneExpression(gene_list=landmark_genes)`. "
        "The synthetic gene names are in no shipped list, so both sides get a pre-selected matrix and the "
        "selection itself is not compared here."
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

    :returns: Parity steps for every variant.
    """
    return sklearn_steps(
        VARIANTS,
        attributes={"n_iter_": "exact", "train_score_": "exact", "validation_score_": "exact"},
        predictions="exact",
        random_state_scope=PARITY,
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
