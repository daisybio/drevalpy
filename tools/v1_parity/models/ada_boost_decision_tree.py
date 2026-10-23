"""AdaBoostDecisionTree parity case: 1.6.1 ``AdaBoostDecisionTree`` against the v2 ``AdaBoostDecisionTree`` preset.

Neither side sets ``random_state``: the boosting rounds draw from the global numpy stream, which is seeded
with the case seed right before training on both sides.

``aligned``
    ``n_estimators=20``, ``max_depth=3``, ``min_samples_split=4``, ``min_samples_leaf=2``.
``defaults``
    The 1.6.1 first grid point (``n_estimators=50``, ``max_depth=4``, ``min_samples_split=2``,
    ``min_samples_leaf=1``) against the unmodified v2 preset; the two are the same configuration.
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

NAME = "AdaBoostDecisionTree"
V1_MODULE, V1_CLASS, V2_PRESET = (
    "drevalpy.models.baselines.sklearn_models",
    "AdaBoostDecisionTree",
    "AdaBoostDecisionTree",
)

VARIANTS: dict[str, dict] = {
    "aligned": {
        "v1": {"n_estimators": 20, "max_depth": 3, "min_samples_split": 4, "min_samples_leaf": 2},
        "v2": {"n_estimators": 20, "max_depth": 3, "min_samples_split": 4, "min_samples_leaf": 2},
    },
    "defaults": {
        "v1": {"n_estimators": 50, "max_depth": 4, "min_samples_split": 2, "min_samples_leaf": 1},
        "v2": {},
    },
}

FINDINGS = [
    (
        "Default hyperparameters agree: 1.6.1 first grid point n_estimators=50, max_depth=4, "
        "min_samples_split=2, min_samples_leaf=1 (grid min_samples_split 2/6, min_samples_leaf 1/3); v2 "
        "declares n_estimators=50, max_depth=4 and the non-tunable min_samples_split=2, min_samples_leaf=1, "
        "learning_rate=1.0 (sklearn's default, which 1.6.1 leaves implicit)."
    ),
    (
        "Neither side pins `random_state`, so the boosting is reproducible only through the global numpy seed. "
        "Reproducing a v2 run requires the caller to seed numpy; B5 may want to pin a `random_state` on both "
        "models."
    ),
    (
        "1.6.1 selects the `landmark_genes` list while loading the CSVs (hyperparameter `gene_list`, a missing "
        "gene raises); the v2 preset consumes the whole gene expression view (`scaledGeneExpression` without "
        "`gene_list`). The case compares a pre-selected matrix; wiring the list into the preset belongs to B5."
    ),
    (
        "Features agree to float32 precision (class forward): v2 holds them as float32, 1.6.1 as float64. The "
        "boosting rounds are identical (same estimator weights and errors) because the trees see the same split "
        "order."
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
        attributes={"feature_importances_": "exact", "estimator_weights_": "exact", "estimator_errors_": "exact"},
        predictions="exact",
        random_state_note=(
            "Both sides leave random_state unset and draw from the global numpy stream (seeded with the case seed)."
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
