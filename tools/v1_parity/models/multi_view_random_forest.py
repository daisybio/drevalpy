"""MultiViewRandomForest parity case: 1.6.1 ``MultiViewRandomForest`` against the v2 ``MultiViewRandomForest`` preset.

Both sides get wide real-symbol omics views (``_omics_common``). 1.6.1 loads them through its own CSV loader and
selects genes with ``view_configs``/``gene_lists``; v2 selects the expression genes with
``scaledGeneExpression(gene_list=...)`` and receives the other views already restricted to their list (its
``raw`` and ``pca`` featurizers have no ``gene_list`` option). 1.6.1 never concatenates the drug features in this
model, so the parity variants give v2 all-zero fingerprints (constant columns, dropped before comparing).

Variants (all hyperparameters explicit; 30 trees, ``max_samples=0.5``, ``max_depth=6``, except ``defaults``):

``raw_expr_mut``
    The 1.6.1 default ``view_configs``: gene_expression -> landmark_genes, mutations ->
    drug_target_genes_all_drugs. v2 `raw` featurizers.
``raw_four_views``
    Expression (landmark), methylation (PCA, 20 components, fit on the unique training samples,
    methylation_intersection), mutations and CNV (drug_target_genes_all_drugs), via top-level
    ``cell_line_views``/``gene_lists``.
``pca_cap``
    ``methylation_n_components=100`` on 32 unique training samples: both sides cap at 32 components (features
    only; the last component spans the null space of the centred data).
``view_configs_n_components``
    Deviation: 1.6.1 ignores ``methylation_n_components`` when the views come from ``view_configs``.
``scaled_expr_mut``
    Deviation: the v2 ``scaledGeneExpression`` against 1.6.1's raw expression.
``with_fingerprints``
    Deviation: 1.6.1 never feeds the drug features; the v2 preset stack does.
``defaults``
    Deviation: 1.6.1 first grid point against the unmodified v2 preset on the wide views.
"""

from __future__ import annotations

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import DEVIATION, StepSpec
from tools.v1_parity.models._omics_common import build_omics_case, omics_steps, run_omics_v1, run_omics_v2

NAME = "MultiViewRandomForest"
V1_MODULE, V1_CLASS = "drevalpy.models.baselines.multi_view_random_forest", "MultiViewRandomForest"

EXPR, MUT, CNV, METH = "gene_expression", "mutations", "copy_number_variation_gistic", "methylation"
DTG = "drug_target_genes_all_drugs"
RF_V1 = {"n_estimators": 30, "criterion": "squared_error", "max_samples": 0.5, "max_depth": 6, "n_jobs": 1}
RF_V2 = {"n_estimators": 30, "max_samples": 0.5, "max_depth": 6, "n_jobs": 1, "random_state": "$seed"}
SCALED = {"scaledGeneExpression": {"gene_list": "landmark_genes"}}


def _config(*sample: object) -> dict:
    return {"sample_featurizer": list(sample), "drug_featurizer": "fingerprints", "predictor": "randomForest"}


def _v1(views: list[str], lists: dict[str, str], **extra: object) -> dict:
    return {**RF_V1, **extra, "view_configs": {"cell_line_views": views, "gene_lists": lists}}


def _v1_plain(views: list[str], lists: dict[str, str], **extra: object) -> dict:
    """Use top-level ``cell_line_views``/``gene_lists``, the only form in which 1.6.1 reads the PCA size."""
    return {**RF_V1, **extra, "cell_line_views": views, "gene_lists": lists}


def _raw(view: str, gene_list: str | None = None) -> object:
    """Return a v2 ``raw`` featurizer entry that selects *gene_list* itself when given."""
    return {f"raw[{view}]": {"gene_list": gene_list}} if gene_list else f"raw[{view}]"


RAW = _raw(EXPR, "landmark_genes")
RAW_MUT = _raw(MUT, DTG)
PAD = {"fingerprints": "fingerprints_zero", "drop_trailing": 32, "v1_pad_columns": 32}

VARIANTS: dict[str, dict] = {
    "raw_expr_mut": {
        "v1": _v1([EXPR, MUT], {EXPR: "landmark_genes", MUT: DTG}),
        "v2_config": _config(RAW, RAW_MUT),
        "v2": RF_V2,
        "v1_scaled_features": True,
        "reference": [(EXPR, "landmark_genes"), (MUT, DTG)],
        **PAD,
    },
    "raw_four_views": {
        "v1": _v1_plain([EXPR, METH, MUT, CNV], {EXPR: "landmark_genes"}, methylation_n_components=20),
        "v2_config": _config(RAW, {f"pca[{METH}]": {"n_components": 20}}, RAW_MUT, _raw(CNV, DTG)),
        "v2": {**RF_V2, "n_components": 20},
        "preselect": {METH: "methylation_intersection"},
        "features": "statistical",
        "reference": [(EXPR, "landmark_genes"), (METH, "methylation_intersection"), (MUT, DTG), (CNV, DTG)],
        **PAD,
    },
    "pca_cap": {
        "v1": _v1_plain([METH], {}, methylation_n_components=100, n_estimators=5),
        "v2_config": _config(f"pca[{METH}]"),
        "v2": {**RF_V2, "n_components": 100, "n_estimators": 5},
        "preselect": {METH: "methylation_intersection"},
        "ignore_last": True,
        "features_only": True,
        "features": "statistical",
        **PAD,
    },
    "view_configs_n_components": {
        "v1": _v1([EXPR, METH], {EXPR: "landmark_genes"}, methylation_n_components=20),
        "v2_config": _config(RAW, {f"pca[{METH}]": {"n_components": 20}}),
        "v2": {**RF_V2, "n_components": 20},
        "preselect": {METH: "methylation_intersection"},
        "features_only": True,
        "scope": DEVIATION,
        "explanation": (
            "1.6.1 MultiViewRandomForest.build_model calls the base build_model before it reads view_configs, so "
            "'methylation' is not yet in cell_line_views when the base class decides whether to read "
            "methylation_n_components: with view_configs the hyperparameter is silently ignored and stays at 100 "
            "(capped at the 32 unique training samples). v2 honours n_components=20."
        ),
        **PAD,
    },
    "scaled_expr_mut": {
        "v1": _v1([EXPR, MUT], {EXPR: "landmark_genes", MUT: DTG}),
        "v1_variant": "raw_expr_mut",
        "v2_config": _config(SCALED, RAW_MUT),
        "v2": RF_V2,
        "scaled_features": "forward",
        "scope": DEVIATION,
        "explanation": (
            "The v2 preset standardizes the expression (arcsinh + scaler fit on the unique training samples), "
            "1.6.1 feeds the raw values to the forest. Both are monotone per gene, so the tree structure is "
            "the same, but the split thresholds are midpoints in a different space and test values between two "
            "training values can fall on the other side."
        ),
        **PAD,
    },
    "with_fingerprints": {
        "v1": _v1([EXPR, MUT], {EXPR: "landmark_genes", MUT: DTG}),
        "v1_variant": "raw_expr_mut",
        "v2_config": _config(RAW, RAW_MUT),
        "v2": RF_V2,
        "scope": DEVIATION,
        "drug_spread": True,
        "explanation": (
            "1.6.1 MultiViewRandomForest.train/predict concatenate cell_line_views only, never the drug views, so "
            "its prediction for a sample is the same for every drug and no fingerprint reaches the forest. The "
            "v2 preset appends the fingerprints, so its forest also splits on drug bits."
        ),
        "spread_explanation": (
            "1.6.1 predicts one value per sample whatever the drug (the drug features are never part of the "
            "design matrix); v2 predicts a drug-specific value."
        ),
    },
    "defaults": {
        "v1": {
            "n_estimators": 100,
            "criterion": "squared_error",
            "max_samples": 0.2,
            "max_depth": 5,
            "n_jobs": 1,
            "view_configs": {
                "cell_line_views": [EXPR, MUT],
                "gene_lists": {EXPR: "landmark_genes", MUT: DTG},
            },
            "methylation_n_components": 100,
        },
        "v2_preset": "MultiViewRandomForest",
        "v2": {"n_jobs": 1},
        "scope": DEVIATION,
        "explanation": (
            "The preset matches the 1.6.1 first grid point in views (expression landmark_genes + "
            "mutations drug_target_genes_all_drugs) and in n_estimators=100, max_samples=0.2, max_depth=5, so the "
            "effective parameters are equal. It still differs in three ways: it appends the fingerprints (1.6.1 "
            "never feeds the drug views here), it scales the expression, and it pins random_state=42."
        ),
    },
}

FINDINGS = [
    (
        "1.6.1 `MultiViewRandomForest.train`/`predict` concatenate `cell_line_views` only; the drug views (the "
        "fingerprints) never reach the forest, so 1.6.1 predicts the same value for a sample for every drug "
        "(drug_spread 0). The v2 `MultiViewRandomForest` preset appends the fingerprints. A harness technique, not "
        "a v2 property: the parity variants give v2 zero fingerprints and pad the 1.6.1 estimator input with the "
        "same number of zero columns so both forests have the same shape (appending constant columns on one side "
        "only changes the sklearn splitter's random stream)."
    ),
    (
        "Gene selection: the 1.6.1 CSV loader and the v2 `raw[...](gene_list=...)` featurizers (`raw` has a "
        "`gene_list` option) select the same columns in the same order; the design matrices agree "
        "to 1e-12, and the predictions and feature importances are exact. `pca[methylation]` still has no "
        "`gene_list` option (the case restricts that view up front), so the 1.6.1 default methylation -> "
        "methylation_intersection cannot be expressed in a recipe."
    ),
    (
        "Expression scaling: 1.6.1 feeds the raw expression to the forest; the v2 preset uses "
        "`scaledGeneExpression` (arcsinh + StandardScaler fit on the unique training samples). The scaled matrix "
        "equals the 1.6.1 matrix after arcsinh + scaling (forward); predictions differ statistically because "
        "trees split at midpoints in a different space."
    ),
    (
        "Methylation PCA: both sides fit on the unique training samples (#479/#482) and cap the components at the "
        "number of unique samples (32 here, 31 compared: the last one spans the centred null space). The v2 PCA "
        "runs in float32 on a float32 view, 1.6.1 in float64, so the PCA features agree only statistically "
        "(about 5e-6 absolute); the resulting forests are identical (same splits, exact predictions)."
    ),
    (
        "1.6.1 bug: `MultiViewRandomForest.build_model` calls the base `build_model` before it reads "
        "`view_configs`, so with `view_configs` the base class does not see `methylation` in the views and "
        "`methylation_n_components` is ignored (stays 100). With top-level `cell_line_views` it is honoured. v2 "
        "always honours `n_components`."
    ),
    (
        "Default hyperparameters (B5): 1.6.1 first grid point n_estimators=100, max_samples=0.2, max_depth=5 "
        "(grid 5/10/30), n_jobs=-1, random_state unset, view_configs = expression(landmark_genes) + "
        "mutations(drug_target_genes_all_drugs), methylation_n_components=100. The v2 preset has the "
        "same views and hyperparameters (effective parameters equal); it adds fingerprints and scales the "
        "expression. The 1.6.1 class default (no view_configs) is four views on whole gene lists, which no "
        "preset expresses."
    ),
    (
        "`random_state`: 1.6.1 leaves it unset (global numpy stream); v2 takes an integer. With the integer "
        "equal to the seed of the global stream the forests are identical."
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

    :returns: Parity steps for the aligned variants, deviation steps for the others.
    """
    return omics_steps(
        VARIANTS,
        attributes={"feature_importances_": "exact"},
        predictions="exact",
        features="exact",
        random_state_note=(
            "1.6.1: unset (global numpy stream, seeded with the case seed). v2: an integer; the case passes the "
            "case seed, which starts the identical stream."
        ),
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
