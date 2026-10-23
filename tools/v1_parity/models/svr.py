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
    preset, which has the same values (``max_iter=500`` is a non-tunable value of the predictor).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import StepSpec
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
    "defaults": {"v1": {"kernel": "rbf", "C": 0.001, "epsilon": 0.001, "max_iter": 500}, "v2": {}},
}

FINDINGS = [
    (
        "Default hyperparameters agree: 1.6.1 first grid point kernel=rbf, C=0.001, epsilon=0.001, max_iter=500 (grid "
        "C 0.001..100, epsilon 0.001..1); v2 declares kernel=rbf, C=0.001, epsilon=0.001 (log-uniform spaces C "
        "1e-3..100, epsilon 1e-3..1) and `max_iter=500` as a non-tunable value. The `defaults` variant runs the "
        "unmodified preset. With C=0.001 the default is close to a constant predictor."
    ),
    (
        "1.6.1 caps libsvm at 500 iterations in every grid point, which stops before convergence on realistic "
        "data (ConvergenceWarning); the v2 predictor caps at 500 as well (`max_iter` is non-tunable there). The "
        "`capped` variant checks that an explicit cap is applied identically."
    ),
    (
        "End to end, the predictions of the converged variants only agree statistically (class statistical): SMO "
        "stops at libsvm's default tol=1e-3, its working-set selection is discontinuous in the kernel values, and "
        "the float32 features of v2 (1e-7 relative) steer it to a different, equally valid solution (different "
        "n_iter_, differences up to the solver tolerance, e.g. 2e-3 absolute for the linear kernel). The "
        "`identical` steps feed both estimators the same float64 matrix and are exact, so the algorithm and "
        "hyperparameters are the same. The `capped` variant (max_iter=500, same path) agrees to float32 "
        "precision, and so does `defaults`: at C=0.001 every training pair is a bound support vector (all dual "
        "coefficients equal C, n_iter_ equal on both sides), which leaves SMO no path to diverge on."
    ),
    (
        "1.6.1 selects the `landmark_genes` list while loading the CSVs (hyperparameter `gene_list`, a missing "
        "gene raises); the v2 preset selects the same list with `scaledGeneExpression(gene_list=landmark_genes)`. "
        "The synthetic gene names are in no shipped list, so both sides get a pre-selected matrix and the "
        "selection itself is not compared here."
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

    :returns: Parity steps for every variant.
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
    # With C=0.001 every training pair of `defaults` is a bound support vector (all dual coefficients equal C), so
    # the dual solution does not depend on the working-set path and only the float32 features move it.
    steady = {"capped/predictions", "defaults/predictions"}
    return [replace(s, tolerance="forward") if s.name in steady else s for s in specs]


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
