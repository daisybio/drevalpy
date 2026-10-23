"""SingleDrugElasticNet parity case: 1.6.1 ``SingleDrugElasticNet`` against the v2 ``SingleDrugElasticNet`` preset.

Variants (``shared_enet`` and ``ragged_enet`` set the hyperparameters explicitly on both sides, ``defaults`` does
not):

``shared_enet``
    ``ElasticNet(alpha=0.1, l1_ratio=0.5)`` on a response matrix in which every drug has the same training
    samples, so the per-drug scaler of 1.6.1 equals the global scaler of v2. The v2 solver settings are the
    preset's own, which are sklearn's defaults as in 1.6.1.
``ragged_enet``
    The same hyperparameters on the standard split, where each drug has its own subset of training samples.
    The scalers differ; recorded as a deviation.
``defaults``
    1.6.1 first grid point (``l1_ratio=0.2``, ``alpha=1``) against the unmodified v2 preset, on the shared split.
"""

from __future__ import annotations

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import DEVIATION, PARITY, StepSpec
from tools.v1_parity.models._single_drug_common import (
    build_single_drug_case,
    run_single_drug_v1,
    run_single_drug_v2,
    single_drug_steps,
)

NAME = "SingleDrugElasticNet"
V1_MODULE, V1_CLASS = "drevalpy.models.baselines.singledrug_baselines", "SingleDrugElasticNet"

VARIANTS: dict[str, dict] = {
    "shared_enet": {
        "pairs": "shared",
        "v1": {"alpha": 0.1, "l1_ratio": 0.5},
        "v2": {"alpha": 0.1, "l1_ratio": 0.5},
    },
    "ragged_enet": {
        "v1": {"alpha": 0.1, "l1_ratio": 0.5},
        "v2": {"alpha": 0.1, "l1_ratio": 0.5},
        "scope": DEVIATION,
        "explanation": "1.6.1 fits the StandardScaler on the training samples of each drug; v2 fits it once on all "
        "training samples (plan 4.4, globally fitted single-drug featurizers). With drug-specific subsets of "
        "samples the scalers, hence the features and the coefficients, differ.",
    },
    "defaults": {"pairs": "shared", "v1": {"alpha": 1.0, "l1_ratio": 0.2}, "v2": {}},
}

FINDINGS = [
    "1.6.1 trains `SingleDrugElasticNet` once per drug on `scale_gene_expression` features whose "
    "`StandardScaler` is fitted on the training samples of that drug; v2 fits `scaledGeneExpression` once over "
    "all training pairs and routes by drug. The two agree (class forward) when every drug has the same training "
    "samples (`shared_enet`), and differ when the drugs were measured on different samples (`ragged_enet`). This "
    "is the intended 'globally fitted single-drug featurizers' deviation of the plan (4.4).",
    "1.6.1 builds `ElasticNet(alpha, l1_ratio)` with sklearn's solver defaults and `random_state` unset; "
    "l1_ratio 0 or 1 select Ridge or Lasso. The v2 predictor pins the same solver settings (max_iter=1000, "
    "tol=1e-4, cyclic, random_state=None; `non_tunable_hyperparameters`), so the effective parameters, including "
    "`random_state`, are equal and the per-drug coefficients agree to float32 precision without any override.",
    "Defaults agree: 1.6.1 first grid point is l1_ratio=0.2, alpha=1 (grids l1_ratio 0.2/0.5/0.9, alpha "
    "1/0.8/0.6/0.4/0.1/0.01/5); the v2 defaults are l1_ratio=0.2, alpha=1.0. The `defaults` variant runs the "
    "unmodified preset.",
    "1.6.1 selects `landmark_genes` while loading; the v2 preset selects the same list with "
    "`scaledGeneExpression(gene_list=landmark_genes)`. The synthetic gene names are in no shipped list, so both "
    "sides get a pre-selected matrix and the selection itself is not compared here.",
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

    :returns: Parity steps for ``shared_enet`` and ``defaults``, deviation steps for ``ragged_enet``.
    """
    return single_drug_steps(
        VARIANTS,
        attributes={"coef_": "forward", "intercept_": "forward"},
        predictions="forward",
        random_state_scope=PARITY,
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
