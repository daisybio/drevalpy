"""SVR parity case: 1.6.1 ``SVR`` against the v2 ``SVR`` zoo preset.

Variants (both sides set ``kernel``, ``C``, ``epsilon`` and ``max_iter`` explicitly):

``rbf``
    ``C=1``, ``epsilon=0.1``, converged (``max_iter=100000``).
``linear``
    The linear kernel.
``capped``
    ``max_iter=500``, the 1.6.1 grid value: libsvm may stop before convergence.
``defaults``
    The 1.6.1 first grid point (``C=0.001``, ``epsilon=0.001``, ``max_iter=500``) against the unmodified v2
    preset (``C=1``, ``epsilon=0.1``, ``max_iter=-1``). Recorded as a deviation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import DEVIATION, StepSpec
from tools.v1_parity.models._sklearn_common import (
    build_sklearn_case,
    run_sklearn_v1,
    run_sklearn_v2,
    sklearn_steps,
)

NAME = "SVR"
V1_MODULE, V1_CLASS, V2_PRESET = "drevalpy.models.baselines.sklearn_models", "SVMRegressor", "SVR"

VARIANTS: dict[str, dict] = {
    "rbf": {
        "v1": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "max_iter": 100000},
        "v2": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "max_iter": 100000},
    },
    "linear": {
        "v1": {"kernel": "linear", "C": 0.1, "epsilon": 0.01, "max_iter": 100000},
        "v2": {"kernel": "linear", "C": 0.1, "epsilon": 0.01, "max_iter": 100000},
    },
    "capped": {
        "v1": {"kernel": "rbf", "C": 10.0, "epsilon": 0.01, "max_iter": 500},
        "v2": {"kernel": "rbf", "C": 10.0, "epsilon": 0.01, "max_iter": 500},
    },
    "defaults": {
        "v1": {"kernel": "rbf", "C": 0.001, "epsilon": 0.001, "max_iter": 500},
        "v2": {},
        "scope": DEVIATION,
        "explanation": (
            "Hyperparameter defaults differ: 1.6.1 first grid point is C=0.001, epsilon=0.001, "
            "max_iter=500; the v2 preset declares C=1.0, epsilon=0.1 and max_iter=-1 (no iteration cap, "
            "not tunable)."
        ),
    },
}

FINDINGS = [
    (
        "Default hyperparameters: 1.6.1 first grid point kernel=rbf, C=0.001, epsilon=0.001, max_iter=500 (grid "
        "C 0.001..100, epsilon 0.001..1); v2 declares kernel=rbf, C=1.0, epsilon=0.1 (log-uniform spaces C "
        "1e-3..100, epsilon 1e-3..1) and `max_iter=-1` as a non-tunable value. The `defaults` variant measures "
        "the effect: with C=0.001 the 1.6.1 default is close to a constant predictor. B5 decides whether to "
        "move the v2 defaults."
    ),
    (
        "1.6.1 caps libsvm at 500 iterations in every grid point, which stops before convergence on realistic "
        "data (ConvergenceWarning); v2 runs to convergence unless `max_iter` is set. The `capped` variant "
        "checks that an explicit cap is applied identically."
    ),
    (
        "End to end, the predictions only agree statistically (class statistical): SMO stops at libsvm's "
        "default tol=1e-3, its working-set selection is discontinuous in the kernel values, and the float32 "
        "features of v2 (1e-7 relative) steer it to a different, equally valid solution (different n_iter_, "
        "differences up to the solver tolerance, e.g. 2e-3 absolute for the linear kernel). The `identical` "
        "steps feed both estimators the same float64 matrix and are exact, so the algorithm and hyperparameters "
        "are the same. The `capped` variant (max_iter=500, same path) agrees to float32 precision."
    ),
    (
        "1.6.1 selects the `landmark_genes` list while loading the CSVs (hyperparameter `gene_list`, a missing "
        "gene raises); the v2 preset consumes the whole gene expression view (`scaledGeneExpression` without "
        "`gene_list`). The case compares a pre-selected matrix; wiring the list into the preset belongs to B5."
    ),
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

    :returns: Parity steps for the aligned variants, deviation steps for the others.
    """
    specs = sklearn_steps(
        VARIANTS,
        attributes={},
        predictions="statistical",
        random_state=False,
        identical=(
            "support_",
            "dual_coef_",
            "intercept_",
        ),
    )
    # `capped` stops both solvers on the same iteration budget, so its path cannot diverge at the tolerance.
    return [replace(s, tolerance="forward") if s.name == "capped/predictions" else s for s in specs]


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
