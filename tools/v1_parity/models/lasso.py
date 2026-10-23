"""Lasso parity case: 1.6.1 ``Lasso`` against the v2 ``Lasso`` zoo preset.

Variants (both sides set ``alpha`` explicitly):

``aligned``
    ``alpha=0.1``; v2 solver settings overridden to the 1.6.1 ones (``max_iter=10000``, ``tol=1e-3``,
    ``selection="random"``) and ``random_state`` equal to the global seed 1.6.1 draws from.
``preset_solver``
    The same 1.6.1 run against the v2 preset's own solver settings (``max_iter=2000``, ``tol=1e-2``,
    ``random_state=0``). Recorded as a deviation.
``defaults``
    1.6.1 first grid point (``alpha=1``) against the unmodified v2 preset. Recorded as a deviation.
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

NAME = "Lasso"
V1_MODULE, V1_CLASS, V2_PRESET = "drevalpy.models.baselines.sklearn_models", "LassoModel", "Lasso"

VARIANTS: dict[str, dict] = {
    "aligned": {
        "v1": {"alpha": 0.1},
        "v2": {"alpha": 0.1, "max_iter": 10000, "tol": 1e-3, "selection": "random", "random_state": "$seed"},
    },
    "preset_solver": {
        "v1": {"alpha": 0.1},
        "v2": {"alpha": 0.1},
        "v1_variant": "aligned",
        "scope": DEVIATION,
        "explanation": "v2 preset solver: max_iter=2000, tol=1e-2, random_state=0; 1.6.1 uses max_iter=10000, "
        "tol=1e-3 and the global numpy stream. Same objective, looser convergence.",
    },
    "defaults": {
        "v1": {"alpha": 1.0},
        "v2": {},
        "scope": DEVIATION,
        "explanation": "1.6.1 first grid point is alpha=1 with max_iter=10000/tol=1e-3; the v2 preset has the same "
        "alpha=1.0 default but max_iter=2000/tol=1e-2/random_state=0. At alpha=1 on standardized "
        "features both fits are the intercept only.",
    },
}

FINDINGS = [
    "1.6.1 builds `Lasso(alpha, max_iter=10000, tol=1e-3, selection='random')` with `random_state` unset (global "
    "numpy stream); the v2 `lasso` predictor pins `max_iter=2000, tol=1e-2, selection='random', random_state=0`. "
    "Every setting is overridable; with the 1.6.1 values and `random_state` equal to the global seed the fitted "
    "coefficients agree to float32 precision (class forward). B5 should set max_iter=10000 and tol=1e-3 if the "
    "1.6.1 solver is to be matched.",
    "The 1.6.1 default (first grid point) and the v2 default are both alpha=1.0, so the default is unchanged "
    "(the 1.6.1 grid is 1, 0.8, 0.6, 0.4, 0.1, 0.01, 5); the v2 search space is log-uniform in [1e-4, 10].",
    "1.6.1 selects the `landmark_genes` list while loading; the v2 `Lasso` preset consumes the whole gene "
    "expression view (`scaledGeneExpression` without `gene_list`). Compared on a pre-selected matrix; the list "
    "wiring belongs to B5.",
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

    :returns: Parity steps for ``aligned``, deviation steps for the other variants.
    """
    return sklearn_steps(
        VARIANTS,
        attributes={"coef_": "forward", "intercept_": "forward"},
        random_state_note="1.6.1 leaves random_state unset (global stream, seeded with the case seed); v2 gets the "
        "same integer, which starts the identical stream, so the random coordinate order is the same.",
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
