"""MultiViewXGBoost parity case: 1.6.1 ``MultiViewXGBoost`` against the v2 ``MultiViewXGBoost`` preset.

Wide real-symbol omics views (``_omics_common``): 1.6.1 loads them through its CSV loader and ``view_configs``;
v2 selects the genes with the ``gene_list`` option of ``scaledGeneExpression`` / ``raw``. Variants are described in
``_multi_view_boosted``. Hyperparameters are explicit on both sides (``n_jobs`` follows each side's own setting;
the harness pins ``OMP_NUM_THREADS=1``).
"""

from __future__ import annotations

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import StepSpec
from tools.v1_parity.models._multi_view_boosted import boosted_variants
from tools.v1_parity.models._omics_common import build_omics_case, omics_steps, run_omics_v1, run_omics_v2

NAME = "MultiViewXGBoost"
V1_MODULE, V1_CLASS = "drevalpy.models.baselines.multi_view_xgboost", "MultiViewXGBoost"

VARIANTS = boosted_variants(
    "xgboost",
    "MultiViewXGBoost",
    (
        "1.6.1 standardizes every methylation column and then fits the PCA on the unique training samples; the v2 "
        "`pca` featurizer has no scaler in front of it, so the components (and the trees grown on them) differ."
    ),
    (
        "xgboost draws the row (`subsample`) and column (`colsample_bytree`) samples from `random_state`, and the "
        "stream changed between xgboost 3.2.0 and 3.4.1: with identical float64 matrices and identical "
        "parameters the 1.6.1 venv (3.4.1, unpinned by 1.6.1: `xgboost>=3.2.0`) and the v2 venv (3.2.0, locked) "
        "fit different trees as soon as either value is below 1, and the same trees when both are 1."
    ),
)

FINDINGS = [
    (
        "Gene selection and scaling: the 1.6.1 CSV loader with `view_configs` and the v2 `scaledGeneExpression` / "
        "`raw[mutations]` `gene_list` options select the same columns in the same order; the standardized "
        "expression agrees to the forward class (v2 scales in float32, 1.6.1 in float64). The 1.6.1 class "
        "default `view_configs` second entry (expression only) is the `expr` variant. The 1.6.1 methylation "
        "preprocessing (StandardScaler + PCA) has no v2 equivalent: `four_views` deviates."
    ),
    (
        "xgboost version: `subsample` and `colsample_bytree` below 1 draw from random_state, and the stream "
        "differs between xgboost 3.2.0 (v2 lock) and 3.4.1 (installed in the 1.6.1 venv; 1.6.1 only requires "
        ">=3.2.0). Identical float64 matrices and parameters give different trees on the two sides, so "
        "`expr_mut_sampling` and `preset` (subsample 0.8, colsample 0.6) are deviations here and `expr_mut` / "
        "`expr` run without sampling. A user moving between versions gets different seeded models."
    ),
    (
        "Default hyperparameters (B5): the 1.6.1 class defaults (no grid) are n_estimators=100, learning_rate=0.1, "
        "max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1.0, random_state=42. The first "
        "grid point of hyperparameters.yaml (what a run uses) is max_depth=10, subsample=0.8, "
        "colsample_bytree=0.6, reg_alpha=0, reg_lambda=0.1 (grid: colsample 0.6/0.8, reg_alpha 0/1), on the "
        "view_configs expression+mutations. The working-tree v2 `xgboost` predictor defaults and the "
        "`MultiViewXGBoost` preset equal that grid point (`preset` variant: identical effective parameters, "
        "identical trees when the xgboost versions agree). reg_lambda is now a public hyperparameter, so every "
        "grid value is expressible; n_jobs is fixed to -1 in v2."
    ),
    (
        "Verified outside the committed run: with xgboost 3.2.0 installed on the 1.6.1 side as well "
        "(`uv pip install --target DIR xgboost==3.2.0`, `PYTHONPATH=DIR`), `expr_mut_sampling` and `preset` give "
        "bitwise identical predictions and feature importances (maximum difference 0.0). The `forward` "
        "tolerance of the importances is for the float32 gain statistics, which differ by 3e-8 between the two "
        "xgboost versions."
    ),
]


def build_case(seed: int) -> Case:
    """Draw the shared inputs and record the variants.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    return build_omics_case(NAME, seed, VARIANTS)


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Parity steps for the aligned variants, deviation steps for the four-view variant.
    """
    return omics_steps(
        VARIANTS,
        attributes={"feature_importances_": "forward"},
        predictions="forward",
        features="forward",
        random_state_note="Both sides pin random_state=42.",
        identical_attributes="forward",
    )


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the 1.6.1 model for every variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    return run_omics_v1(case, V1_MODULE, V1_CLASS)


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the v2 model for every variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    return run_omics_v2(case, NAME)
