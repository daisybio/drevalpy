"""Lasso parity case: 1.6.1 ``Lasso`` against the v2 ``Lasso`` zoo preset.

Both sides run the preset's own solver settings (``max_iter=10000``, ``tol=1e-3``, ``selection="random"``).
1.6.1 leaves ``random_state`` unset and draws from the global numpy stream; the v2 preset pins
``random_state=0``. Seeding the global stream with 0 right before 1.6.1 trains (``np_seed``) starts the identical
stream, so the random coordinate order is the same. No variant overrides a v2 solver setting.

``preset``
    ``alpha=0.1`` on both sides, v2 otherwise unmodified.
``defaults``
    1.6.1 first grid point (``alpha=1``) against the unmodified v2 preset. At ``alpha=1`` on standardized
    features both fits are the intercept only, so this variant checks the defaults and the intercept, not the
    coordinate descent.
"""

from __future__ import annotations

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import StepSpec
from tools.v1_parity.models._sklearn_common import (
    build_sklearn_case,
    run_sklearn_v1,
    run_sklearn_v2,
    sklearn_steps,
)

NAME = "Lasso"
V1_MODULE, V1_CLASS, V2_PRESET = "drevalpy.models.baselines.sklearn_models", "LassoModel", "Lasso"

VARIANTS: dict[str, dict] = {
    "preset": {"v1": {"alpha": 0.1}, "v2": {"alpha": 0.1}, "np_seed": 0},
    "defaults": {"v1": {"alpha": 1.0}, "v2": {}, "np_seed": 0},
}

FINDINGS = [
    "1.6.1 builds `Lasso(alpha, max_iter=10000, tol=1e-3, selection='random')` with `random_state` unset (global "
    "numpy stream). The v2 `lasso` predictor has the same solver settings (`non_tunable_hyperparameters`) and "
    "pins `random_state=0`, which makes a v2 fit reproducible where 1.6.1 draws a different coordinate order per "
    "process. With the global stream seeded with 0 on the 1.6.1 side the random coordinate order is identical and "
    "the fitted coefficients agree to float32 precision (class forward) with the unmodified preset.",
    "The 1.6.1 default (first grid point) and the v2 default are both alpha=1.0 (the 1.6.1 grid is 1, 0.8, 0.6, "
    "0.4, 0.1, 0.01, 5); the v2 search space is log-uniform in [1e-4, 10].",
    "1.6.1 selects the `landmark_genes` list while loading; the v2 `Lasso` preset selects the same list with "
    "`scaledGeneExpression(gene_list=landmark_genes)`. The synthetic gene names are in no shipped list, so both "
    "sides get a pre-selected matrix and the selection itself is not compared here.",
    "Features agree to float32 precision (class forward): v2 holds them as float32, 1.6.1 as float64.",
]


def build_case(seed: int) -> Case:
    """Draw the shared inputs and record the variants.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    return build_sklearn_case(NAME, seed, VARIANTS, n_samples=60, n_drugs=8, n_genes=50, n_bits=32)


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Parity steps for every variant, the ``random_state`` steps as deviations.
    """
    return sklearn_steps(
        VARIANTS,
        attributes={"coef_": "forward", "intercept_": "forward"},
        random_state_note="1.6.1 leaves random_state unset (global stream, seeded with 0 here); the v2 preset pins "
        "random_state=0, which starts the identical stream, so the random coordinate order is the same.",
    )


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the 1.6.1 ``Lasso`` model for every variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    return run_sklearn_v1(case, V1_MODULE, V1_CLASS)


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the v2 ``Lasso`` zoo preset for every variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    return run_sklearn_v2(case, V2_PRESET)
