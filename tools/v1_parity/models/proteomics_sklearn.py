"""Proteomics sklearn parity case: 1.6.1 ``cell_line_views: proteomics`` against v2 ``normalizedProteomics``.

1.6.1 runs ``ElasticNetModel`` / ``RandomForest`` with ``cell_line_views = "proteomics"``: the whole CSV is loaded
(``load_generic_csv``, no gene list), ``prepare_proteomics`` applies ``log10`` (zeros -> NaN) and the
``ProteomicsMedianCenterAndImputeTransformer`` (fit on the unique training samples; no scaler), and the
fingerprints are appended. v2 is the recipe ``normalizedProteomics:fingerprints:<predictor>`` on the same view.
The view has 120 proteins with 0-60 % missing values, so with 32 unique training samples roughly half of the
proteins pass the 0.7 completeness threshold.

Variants (the ``proteomics_*`` hyperparameters are explicit on both sides unless noted):

``enet_threshold``
    ``proteomics_n_features=20``, below the number of complete proteins: the ``completeness >= required``
    branch. ElasticNet (``alpha=0.1``, ``l1_ratio=0.5``, sklearn's solver defaults on the v2 side).
``enet_fallback``
    ``proteomics_n_features=100``, above it: the ``argsort(completeness)[::-1][:n]`` branch, whose result
    depends on how the sort orders ties and so on the numpy version (2.5.4 against 2.4.6 here).
``rf_threshold``
    RandomForest with ``proteomics_n_features=20``.
``rf_defaults``
    RandomForest with the ``proteomics_*`` hyperparameters left at their defaults (``n_features=1000`` exceeds the
    120 proteins, so every protein is kept, in ``argsort`` order).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import DEVIATION, StepSpec
from tools.v1_parity.models._omics_common import build_omics_case, omics_steps, run_omics_v1, run_omics_v2

NAME = "ProteomicsSklearn"
V1_MODULE, V1_CLASS = "drevalpy.models.baselines.sklearn_models", "ElasticNetModel"

PROTEOMICS = {
    "proteomics_feature_threshold": 0.7,
    "proteomics_normalization_width": 0.3,
    "proteomics_normalization_downshift": 1.8,
}
SOLVER = {"max_iter": 1000, "tol": 1e-4, "selection": "cyclic"}
ENET = {"alpha": 0.1, "l1_ratio": 0.5}
RF = {"n_estimators": 30, "max_samples": 0.5, "max_depth": 6, "n_jobs": 1}
REFERENCE = {"reference": [("proteomics", None)], "reference_fingerprints": True}
V1_VIEWS = {"cell_line_views": "proteomics", "drug_views": "fingerprints"}


def _enet(n_features: int) -> dict:
    return {
        "v1": {**V1_VIEWS, **ENET, **PROTEOMICS, "proteomics_n_features": n_features},
        "v2_preset": "normalizedProteomics:fingerprints:elasticNet",
        "v2": {**ENET, **SOLVER, **PROTEOMICS, "proteomics_n_features": n_features},
        **REFERENCE,
    }


def _forest(**proteomics: float) -> dict:
    return {
        "v1_class": "RandomForest",
        "v1": {**V1_VIEWS, **RF, "criterion": "squared_error", **proteomics},
        "v2_preset": "normalizedProteomics:fingerprints:randomForest",
        "v2": {**RF, "criterion": "squared_error", "random_state": "$seed", **proteomics},
        **REFERENCE,
    }


VARIANTS: dict[str, dict] = {
    "enet_threshold": _enet(20),
    "enet_fallback": _enet(100),
    "rf_threshold": _forest(**PROTEOMICS, proteomics_n_features=20),
    "rf_defaults": _forest(),
}

FLOAT32_SENSITIVE = {"rf_threshold/fitted_feature_importances_", "rf_threshold/predictions"}
FLOAT32_SPLITS = (
    "v2 views are float32, so the proteomics design matrix differs from 1.6.1's float64 one by up to 1.3e-6, which "
    "flips a few forest split decisions. Refitted on one float64 matrix (`identical/*`) the forests are bitwise equal."
)

FINDINGS = [
    (
        "1.6.1 single-view sklearn models load the proteomics CSV whole (`load_generic_csv`); the "
        "`drug_target_genes_all_drugs_proteomics` default exists only in the multi-view loader. v2 "
        "`normalizedProteomics` also consumes the whole view, so the two agree. v2 has no `gene_list` option "
        "for proteomics, which the 1.6.1 multi-view models (`view_configs` / `gene_lists`) would need."
    ),
    (
        "The transformer is the same code on both sides: protein selection, the median-centring factor and the "
        "seeded imputation agree. v2 computes `log10` and the centring in float32 (the view is float32), 1.6.1 in "
        "float64, so the design matrices agree to the forward class, not bitwise."
    ),
    (
        "Deviation (`rf_threshold`, float32 features): the proteomics design matrices differ by up to "
        "1.3e-6 absolute (2.4e-7 relative; the v2 featurizer log10-transforms, centres and imputes in "
        "float32), which is enough to flip a few split decisions in the forest: predictions differ by up "
        "to 0.05 (Pearson 0.99988) and the importances by 1e-3, so the declared `forward` class fails "
        "(achieved: statistical). The estimator is not the cause: refitted on one float64 matrix "
        "(`identical/*`) both forests are bitwise equal, and `rf_defaults` (all proteins kept) and both "
        "ElasticNet variants agree at the forward class or better. The declared class is kept; the two "
        "steps are recorded as a deviation, because float32 views are a documented v2 difference."
    ),
    (
        "`ProteomicsMedianCenterAndImputeTransformer.fit` breaks completeness ties with `np.argsort(...)[::-1]` "
        "(unstable sort, `n_features` above the number of complete proteins). `enet_fallback` and "
        "`rf_defaults` exercise it with numpy 2.5.4 (1.6.1) against 2.4.6 (v2): the selected proteins and their "
        "order are compared through the design matrices."
    ),
    (
        "Defaults: `proteomics_feature_threshold=0.7`, `proteomics_n_features=1000`, "
        "`proteomics_normalization_width=0.3`, `proteomics_normalization_downshift=1.8` are the same in 1.6.1 and "
        "v2. 1.6.1 ElasticNet uses sklearn's solver defaults (max_iter=1000, tol=1e-4, cyclic) and the v2 "
        "predictor pins other values (see the ElasticNet case); the case sets the sklearn values on the v2 side. "
        "The 1.6.1 default RandomForest grid point (n_estimators=100, max_samples=0.2, max_depth=5) equals the "
        "working-tree v2 `randomForest` defaults."
    ),
    (
        "`random_state`: 1.6.1 leaves it unset (global numpy stream, seeded with the case seed); v2 takes an "
        "integer. With the integer equal to that seed the forests are identical."
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

    :returns: Parity steps for every variant.
    """
    enet = {k: v for k, v in VARIANTS.items() if k.startswith("enet")}
    forest = {k: v for k, v in VARIANTS.items() if k.startswith("rf")}
    note = "1.6.1: unset (global numpy stream seeded with the case seed). v2: the case seed."
    specs = [
        *omics_steps(
            enet,
            attributes={"coef_": "forward", "intercept_": "forward"},
            predictions="forward",
            features="forward",
            random_state_note=note,
        ),
        *omics_steps(
            forest,
            attributes={"feature_importances_": "forward"},
            predictions="forward",
            features="forward",
            random_state_note=note,
        ),
    ]
    return [
        replace(s, scope=DEVIATION, explanation=FLOAT32_SPLITS) if s.name in FLOAT32_SENSITIVE else s for s in specs
    ]


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the 1.6.1 models for every variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    return run_omics_v1(case, V1_MODULE, V1_CLASS)


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the v2 models for every variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    return run_omics_v2(case, NAME)
