"""SingleDrugRandomForest parity case: 1.6.1 ``SingleDrugRandomForest`` against the v2 preset of the same name.

Variants (``shared_forest`` and ``ragged_forest`` set the hyperparameters explicitly on both sides, ``defaults``
does not):

``shared_forest``
    A small forest (30 trees, ``max_samples=0.5``, ``max_depth=6``, ``n_jobs=1``) on a response matrix in which
    every drug has the same training samples. 1.6.1 leaves ``random_state`` unset and draws from the global
    numpy stream, seeded before every drug; v2 gets that integer as ``random_state``, so each drug's forest
    starts from the same stream.
``ragged_forest``
    The same hyperparameters on the standard split, where each drug has its own subset of training samples.
    The scalers differ; recorded as a deviation.
``defaults``
    The 1.6.1 first grid point (``n_estimators=100``, ``max_samples=0.2``, ``max_depth=5``, ``n_jobs=-1``)
    against the unmodified v2 preset on the shared split. The v2 preset pins ``random_state=42``; the global numpy
    stream of 1.6.1 is seeded with 42 before every drug (``np_seed``), which starts the identical stream.
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

NAME = "SingleDrugRandomForest"
V1_MODULE, V1_CLASS = "drevalpy.models.baselines.singledrug_baselines", "SingleDrugRandomForest"

FOREST_V1 = {"n_estimators": 30, "criterion": "squared_error", "max_samples": 0.5, "max_depth": 6, "n_jobs": 1}
FOREST_V2 = {**FOREST_V1, "random_state": "$seed"}

VARIANTS: dict[str, dict] = {
    "shared_forest": {"pairs": "shared", "v1": FOREST_V1, "v2": FOREST_V2},
    "ragged_forest": {
        "v1": FOREST_V1,
        "v2": FOREST_V2,
        "scope": DEVIATION,
        "explanation": "1.6.1 fits the StandardScaler on the training samples of each drug; v2 fits it once on all "
        "training samples (plan 4.4, globally fitted single-drug featurizers). With drug-specific subsets of "
        "samples the features differ. A forest is invariant to per-column scaling, so the predictions can "
        "still agree closely.",
    },
    "defaults": {
        "pairs": "shared",
        "v1": {"n_estimators": 100, "criterion": "squared_error", "max_samples": 0.2, "max_depth": 5, "n_jobs": -1},
        "v2": {},
        "np_seed": 42,
    },
}

FINDINGS = [
    "1.6.1 trains `SingleDrugRandomForest` once per drug with `random_state` unset (global numpy stream); v2 "
    "pins `random_state=42` in the predictor and builds one forest per drug from it. With `random_state` equal to "
    "the integer the global stream is seeded with before each drug, the per-drug forests are identical "
    "(feature importances and predictions exact).",
    "Same scaler deviation as `SingleDrugElasticNet`: 1.6.1 standardizes with the training samples of the drug, "
    "v2 with all training samples (plan 4.4). With a shared sample set the features agree to float32 precision.",
    "Defaults agree: 1.6.1 first grid point n_estimators=100, max_samples=0.2, max_depth=5 (grid 5/10/30); v2 "
    "declares n_estimators=100, max_samples=0.2, max_depth=5, n_jobs=-1. Only `random_state` differs (unset vs "
    "42). The `defaults` variant runs the unmodified preset against 1.6.1 with its global stream seeded with 42.",
    "1.6.1 selects `landmark_genes` while loading; the v2 preset selects the same list with "
    "`scaledGeneExpression(gene_list=landmark_genes)`. The synthetic gene names are in no shipped list, so both "
    "sides get a pre-selected matrix and the selection itself is not compared here.",
]


def build_case(seed: int) -> Case:
    """Draw the shared inputs and record the variants.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    return build_single_drug_case(NAME, seed, VARIANTS)


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Parity steps for ``shared_forest`` and ``defaults``, deviation steps for ``ragged_forest``.
    """
    return single_drug_steps(
        VARIANTS,
        attributes={"feature_importances_": "exact"},
        predictions="exact",
        random_state_note="1.6.1 leaves random_state unset (global stream, seeded with the case seed or, in "
        "`defaults`, with 42); v2 gets an integer (the case seed in `shared_forest`, the preset's 42 in `defaults`), "
        "which starts the identical stream.",
    )


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Train and predict every drug with the 1.6.1 ``SingleDrugRandomForest`` model.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<drug>/<quantity>``.
    """
    return run_single_drug_v1(case, V1_MODULE, V1_CLASS)


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Train and predict every drug with the v2 ``SingleDrugRandomForest`` zoo preset.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<drug>/<quantity>``.
    """
    from drevalpy.components.predictors.sklearn_models import SingleDrugRandomForestPredictor

    return run_single_drug_v2(case, NAME, SingleDrugRandomForestPredictor)
