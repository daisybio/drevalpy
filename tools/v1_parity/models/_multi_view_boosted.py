"""Variants shared by the ``MultiViewXGBoost`` and ``MultiViewLightGBM`` parity cases (no case module of its own).

Both 1.6.1 models build their design matrix as ``cell_line_views + drug_views`` (the fingerprints), apply
``arcsinh`` + a ``StandardScaler`` to the expression and, for methylation, a ``StandardScaler`` followed by a PCA
(both fit on the unique training samples). The v2 stack is ``scaledGeneExpression`` + ``raw[...]`` (+
``pca[methylation]``) + ``fingerprints`` + the ``xgboost`` / ``lightgbm`` predictor.

Variants (``kind`` selects the library-specific hyperparameters):

``expr_mut`` / ``expr``
    The two 1.6.1 default ``view_configs``: expression (landmark_genes) with and without mutations
    (drug_target_genes_all_drugs), explicit non-default hyperparameters. Parity.
``preset``
    The 1.6.1 first grid point of ``hyperparameters.yaml`` against the unmodified v2 preset, whose defaults equal
    that grid point. Parity, unless the library draws row or column samples differently on the two sides
    (``sampling_explanation``).
``four_views``
    The 1.6.1 class default views (expression landmark_genes, methylation, mutations, CNV) against v2 with
    ``pca[methylation]``. Deviation: 1.6.1 scales the methylation columns before the PCA, v2 has no such step.
"""

from __future__ import annotations

from typing import Any

from tools.v1_parity.compare import DEVIATION

EXPR, MUT, CNV, METH = "gene_expression", "mutations", "copy_number_variation_gistic", "methylation"
DTG = "drug_target_genes_all_drugs"
VIEW_EXPR_MUT = {"cell_line_views": [EXPR, MUT], "gene_lists": {EXPR: "landmark_genes", MUT: DTG}}
VIEW_EXPR = {"cell_line_views": [EXPR], "gene_lists": {EXPR: "landmark_genes"}}
SCALED = {"scaledGeneExpression": {"gene_list": "landmark_genes"}}
RAW_MUT = {f"raw[{MUT}]": {"gene_list": DTG}}

#: Non-default values of every public knob, per library (the same keys on both sides).
EXPLICIT = {
    "xgboost": {
        "n_estimators": 30,
        "learning_rate": 0.15,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "reg_alpha": 0.5,
        "reg_lambda": 0.3,
    },
    "lightgbm": {
        "n_estimators": 30,
        "learning_rate": 0.15,
        "max_depth": 4,
        "num_leaves": 15,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "reg_alpha": 0.5,
        "reg_lambda": 0.3,
    },
}
#: The first grid point of the 1.6.1 ``hyperparameters.yaml`` (keys the file does not give are the code defaults).
GRID_POINT = {
    "xgboost": {
        "learning_rate": 0.1,
        "max_depth": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.0,
        "reg_lambda": 0.1,
    },
    "lightgbm": {
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
}


def _config(*sample: object, predictor: str) -> dict[str, Any]:
    return {"sample_featurizer": list(sample), "drug_featurizer": "fingerprints", "predictor": predictor}


def boosted_variants(
    kind: str, preset: str, four_views_explanation: str, sampling_explanation: str = ""
) -> dict[str, dict[str, Any]]:
    """Build the variants of one library.

    :param kind: ``"xgboost"`` or ``"lightgbm"`` (the v2 predictor name).
    :param preset: Name of the v2 zoo preset.
    :param four_views_explanation: Why the four-view variant deviates.
    :param sampling_explanation: When given, the variants that draw row or column samples (``subsample`` or
        ``colsample_bytree`` below 1) are deviations with this explanation, and ``expr_mut`` / ``expr`` run
        without sampling instead.
    :returns: Variant settings for :func:`tools.v1_parity.models._omics_common.build_omics_case`.
    """
    sampled = EXPLICIT[kind]
    hp = {**sampled, "subsample": 1.0, "colsample_bytree": 1.0} if sampling_explanation else sampled
    scope = {"scope": DEVIATION, "explanation": sampling_explanation} if sampling_explanation else {}
    reference = [(EXPR, "landmark_genes"), (MUT, DTG)]
    variants: dict[str, dict[str, Any]] = {
        "expr_mut": {
            "v1": {**hp, "view_configs": VIEW_EXPR_MUT},
            "v2_config": _config(SCALED, RAW_MUT, predictor=kind),
            "v2": hp,
            "reference": reference,
            "reference_fingerprints": True,
        },
        "expr": {
            "v1": {**hp, "view_configs": VIEW_EXPR},
            "v2_config": _config(SCALED, predictor=kind),
            "v2": hp,
            "reference": reference[:1],
            "reference_fingerprints": True,
        },
    }
    if sampling_explanation:
        variants["expr_mut_sampling"] = {
            "v1": {**sampled, "view_configs": VIEW_EXPR_MUT},
            "v2_config": _config(SCALED, RAW_MUT, predictor=kind),
            "v2": sampled,
            **scope,
        }
    variants["preset"] = {
        "v1": {**GRID_POINT[kind], "view_configs": VIEW_EXPR_MUT},
        "v2_preset": preset,
        "v2": {},
        "reference": reference,
        "reference_fingerprints": True,
        **scope,
    }
    variants["four_views"] = {
        "v1": {**hp, "cell_line_views": [EXPR, METH, MUT, CNV], "gene_lists": {EXPR: "landmark_genes"}}
        | {"methylation_n_components": 20},
        "v2_config": _config(
            SCALED,
            {f"pca[{METH}]": {"n_components": 20}},
            RAW_MUT,
            {f"raw[{CNV}]": {"gene_list": DTG}},
            predictor=kind,
        ),
        "v2": {**hp, "n_components": 20},
        "preselect": {METH: "methylation_intersection"},
        "scope": DEVIATION,
        "explanation": four_views_explanation,
    }
    return variants
