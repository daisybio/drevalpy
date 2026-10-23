"""MultiViewLightGBM parity case: 1.6.1 ``MultiViewLightGBM`` against the v2 ``MultiViewLightGBM`` preset.

Wide real-symbol omics views (``_omics_common``): 1.6.1 loads them through its CSV loader and ``view_configs``;
v2 selects the genes with the ``gene_list`` option of ``scaledGeneExpression`` / ``raw``. Variants are described in
``_multi_view_boosted``. Hyperparameters are explicit on both sides (``n_jobs`` follows each side's own setting;
the harness pins ``OMP_NUM_THREADS=1``).
"""

from __future__ import annotations

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import PARITY, StepSpec
from tools.v1_parity.models._multi_view_boosted import boosted_variants
from tools.v1_parity.models._omics_common import build_omics_case, omics_steps, run_omics_v1, run_omics_v2

NAME = "MultiViewLightGBM"
V1_MODULE, V1_CLASS = "drevalpy.models.baselines.multi_view_lightgbm", "MultiViewLightGBM"

VARIANTS = boosted_variants(
    "lightgbm",
    "MultiViewLightGBM",
    (
        "1.6.1 standardizes every methylation column and then fits the PCA on the unique training samples; the v2 "
        "`pca` featurizer has no scaler in front of it, so the components (and the trees grown on them) differ."
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
        "Default hyperparameters (B5): the 1.6.1 class defaults are n_estimators=100, learning_rate=0.1, "
        "max_depth=6, num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=0, "
        "random_state=42, n_jobs=-1. The first grid point of hyperparameters.yaml (what a run uses) is "
        "num_leaves=31, subsample=0.8, colsample_bytree=0.6, reg_alpha=0, reg_lambda=0 (max_depth is not in the "
        "grid, so 6), on the view_configs expression+mutations. The v2 `lightgbm` predictor "
        "defaults and the `MultiViewLightGBM` preset equal that grid point (`preset` variant: identical "
        "effective parameters, exact predictions). Both sides pin random_state=42 (a parity step)."
    ),
    (
        "`subsample` has no effect in LightGBM unless `subsample_freq` > 0, and neither side sets it, so "
        "bagging is never active on either side (the case sets subsample=0.7 and the effective parameters show "
        "subsample_freq=0 on both). `colsample_bytree` (feature_fraction) is active and gives identical trees. "
        "With the same lightgbm version (4.7.0) on both sides the predictions are exact."
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
        random_state_scope=PARITY,
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
