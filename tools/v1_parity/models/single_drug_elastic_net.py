"""SingleDrugElasticNet parity case: 1.6.1 ``SingleDrugElasticNet`` against the v2 ``SingleDrugElasticNet`` preset.

Variants (both sides set the hyperparameters explicitly):

``shared_enet``
    ``ElasticNet(alpha=0.1, l1_ratio=0.5)`` with sklearn's solver defaults (the 1.6.1 behaviour) set on the v2
    side, on a response matrix in which every drug has the same training samples, so the per-drug scaler of
    1.6.1 equals the global scaler of v2.
``ragged_enet``
    The same hyperparameters on the standard split, where each drug has its own subset of training samples.
    The scalers differ; recorded as a deviation.
``shared_preset_solver``
    The 1.6.1 run of ``shared_enet`` against the v2 preset's own solver settings. Recorded as a deviation.
``defaults``
    1.6.1 first grid point (``l1_ratio=0.2``, ``alpha=1``) against the unmodified v2 preset. Deviation.
"""

from __future__ import annotations

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import DEVIATION, StepSpec
from tools.v1_parity.models._single_drug_common import (
    build_single_drug_case,
    run_single_drug_v1,
    run_single_drug_v2,
    single_drug_steps,
)

NAME = "SingleDrugElasticNet"
V1_MODULE, V1_CLASS = "drevalpy.models.baselines.singledrug_baselines", "SingleDrugElasticNet"

SKLEARN_SOLVER = {"max_iter": 1000, "tol": 1e-4, "selection": "cyclic"}

VARIANTS: dict[str, dict] = {
    "shared_enet": {
        "pairs": "shared",
        "v1": {"alpha": 0.1, "l1_ratio": 0.5},
        "v2": {"alpha": 0.1, "l1_ratio": 0.5, **SKLEARN_SOLVER},
    },
    "ragged_enet": {
        "v1": {"alpha": 0.1, "l1_ratio": 0.5},
        "v2": {"alpha": 0.1, "l1_ratio": 0.5, **SKLEARN_SOLVER},
        "scope": DEVIATION,
        "explanation": "1.6.1 fits the StandardScaler on the training samples of each drug; v2 fits it once on all "
        "training samples (plan 4.4, globally fitted single-drug featurizers). With drug-specific subsets of "
        "samples the scalers, hence the features and the coefficients, differ.",
    },
    "shared_preset_solver": {
        "pairs": "shared",
        "v1": {"alpha": 0.1, "l1_ratio": 0.5},
        "v2": {"alpha": 0.1, "l1_ratio": 0.5},
        "scope": DEVIATION,
        "explanation": "v2 preset solver: max_iter=2000, tol=1e-2, random coordinate order; 1.6.1 uses sklearn's "
        "defaults. Same objective, looser convergence.",
    },
    "defaults": {
        "pairs": "shared",
        "v1": {"alpha": 1.0, "l1_ratio": 0.2},
        "v2": {},
        "scope": DEVIATION,
        "explanation": "1.6.1 first grid point is l1_ratio=0.2, alpha=1 (grid l1_ratio 0.2/0.5/0.9); the v2 preset "
        "default is l1_ratio=0.5, alpha=0.1, plus its own solver settings.",
    },
}

FINDINGS = [
    "1.6.1 trains `SingleDrugElasticNet` once per drug on `scale_gene_expression` features whose "
    "`StandardScaler` is fitted on the training samples of that drug; v2 fits `scaledGeneExpression` once over "
    "all training pairs and routes by drug. The two agree (class forward) when every drug has the same training "
    "samples (`shared_enet`), and differ when the drugs were measured on different samples (`ragged_enet`). This "
    "is the intended 'globally fitted single-drug featurizers' deviation of the plan (4.4).",
    "1.6.1 builds `ElasticNet(alpha, l1_ratio)` with sklearn's solver defaults; l1_ratio 0 or 1 select Ridge or "
    "Lasso. v2 pins max_iter=2000, tol=1e-2, selection='random', random_state=0 (overridable). With sklearn's "
    "values set explicitly the per-drug coefficients agree to float32 precision.",
    "Defaults: 1.6.1 first grid point is l1_ratio=0.2, alpha=1 (grids l1_ratio 0.2/0.5/0.9, alpha "
    "1/0.8/0.6/0.4/0.1/0.01/5); v2 defaults are l1_ratio=0.5, alpha=0.1. B5 should decide whether to match.",
    "The v2 preset uses the whole gene view (`scaledGeneExpression` without `gene_list`) while 1.6.1 selects "
    "`landmark_genes`. Compared on a pre-selected matrix; the list wiring belongs to B5.",
    "v2 features are float32; the `identical` steps feed both estimators the same float64 matrix and are exact.",
]


def build_case(seed: int) -> Case:
    """Draw the shared inputs and record the variants.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    return build_single_drug_case(NAME, seed, VARIANTS)


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Parity steps for ``shared_enet``, deviation steps for the others.
    """
    return single_drug_steps(
        VARIANTS,
        attributes={"coef_": "forward", "intercept_": "forward"},
        predictions="forward",
        random_state_note="1.6.1 leaves random_state unset; v2 passes the case seed (irrelevant for the cyclic "
        "solver), so the recorded values differ.",
    )


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Train and predict every drug with the 1.6.1 ``SingleDrugElasticNet`` model.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<drug>/<quantity>``.
    """
    return run_single_drug_v1(case, V1_MODULE, V1_CLASS)


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Train and predict every drug with the v2 ``SingleDrugElasticNet`` zoo preset.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<drug>/<quantity>``.
    """
    from drevalpy.components.predictors.sklearn_models import SingleDrugElasticNetPredictor

    return run_single_drug_v2(case, NAME, SingleDrugElasticNetPredictor)
