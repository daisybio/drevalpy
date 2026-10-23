"""ElasticNet parity case: 1.6.1 ``ElasticNet`` against the v2 ``ElasticNet`` zoo preset.

The estimator input is compared, not the gene selection: 1.6.1 selects ``landmark_genes`` while
loading the CSVs and v2's ``scaledGeneExpression`` preset consumes the whole view (see
``FINDINGS``), so both sides receive the same, already selected, gene matrix.

Variants (``ridge`` is the first set of the 1.6.1 ``hyperparameters.yaml`` grid, ``l1_ratio=0``,
``alpha=1``, which 1.6.1 builds as ``Ridge``):

``ridge``
    1.6.1 ``Ridge(alpha=1)`` against v2 ``elasticNet`` with ``l1_ratio=0`` (also ``Ridge``).
``enet``
    1.6.1 ``ElasticNet(alpha=0.1, l1_ratio=0.5)`` (sklearn solver defaults) against v2 with its
    solver settings overridden to those defaults.
``enet_preset``
    The same 1.6.1 run against v2 with the preset's own solver settings (``tol=1e-2``, random
    coordinate order). Recorded as a deviation.

v2 holds features as float32 end to end while 1.6.1 computes in float64, so no class stricter
than ``forward`` holds for the features, and ``Ridge`` (solved in float32 by v2) needs ``train``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, record_calls, tabular_case_arrays
from tools.v1_parity.compare import DEVIATION, PARITY, StepSpec

NAME = "ElasticNet"

#: sklearn's own defaults, which 1.6.1 relies on (it passes only alpha and l1_ratio).
SKLEARN_SOLVER = {"max_iter": 1000, "tol": 1e-4, "selection": "cyclic"}

VARIANTS: dict[str, dict[str, Any]] = {
    "ridge": {"l1_ratio": 0.0, "alpha": 1.0, "v2_extra": {}},
    "enet": {"l1_ratio": 0.5, "alpha": 0.1, "v2_extra": SKLEARN_SOLVER},
    "enet_preset": {"l1_ratio": 0.5, "alpha": 0.1, "v2_extra": {}, "v1_variant": "enet"},
}

FINDINGS = [
    "1.6.1 selects the `landmark_genes` list while loading (hyperparameter `gene_list`; a missing gene "
    "raises). The v2 `ElasticNet` preset uses `scaledGeneExpression`, which has no gene-list option and "
    "scales the whole view, so the preset trains on every gene of the dataset. Compared at the "
    "estimator-input level on a pre-selected matrix. To be fixed by packages A1/B5.",
    "v2 holds features as float32 end to end (`DenseViewFeaturizer` casts to float32, so the estimator "
    "receives a float32 matrix); 1.6.1 computes in float64. Features therefore agree to ~1e-7 relative "
    "(class forward), not exactly. Ridge solves in float32 in v2: refitting sklearn Ridge on the same "
    "matrix in float32 vs float64 differs by 2.4e-5 absolute (predictions ~7), which is the whole Ridge "
    "difference (class train).",
    "1.6.1 builds `sklearn.ElasticNet(alpha, l1_ratio)` with sklearn's solver defaults "
    "(max_iter=1000, tol=1e-4, cyclic); v2 pins max_iter=2000, tol=1e-2, selection='random', "
    "random_state=0. All are overridable in v2, and with the sklearn values the coefficients agree to "
    "6e-8 (class forward). With the preset values they only agree statistically (deviation steps).",
    "The first 1.6.1 hyperparameter set (l1_ratio=0, alpha=1) is `Ridge`; v2 also maps l1_ratio=0 to "
    "`Ridge`, so both estimator types agree.",
    "v2 changed the default alpha to 0.1 (the 1.6.1 grid starts at 1); the case sets alpha explicitly on both sides.",
]


def build_case(seed: int) -> Case:
    """Draw the shared tabular inputs and record the variants.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    arrays = tabular_case_arrays(seed, n_samples=30, n_drugs=6, n_genes=40, n_bits=32)
    meta = {"variants": VARIANTS, "gene_list": "pre-selected: all 40 synthetic genes"}
    return Case(model=NAME, seed=seed, arrays=arrays, meta=meta)


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Parity steps for ``ridge`` and ``enet``, deviation steps for ``enet_preset``.
    """
    solver = "v2 preset solver: tol=1e-2, random coordinate order; 1.6.1 uses sklearn's defaults."
    ridge_params = (
        "v2 passes max_iter=2000, tol=1e-2 to Ridge; both are inert for the dense 'auto' (cholesky) "
        "solver 1.6.1 also uses, which the matching predictions confirm."
    )
    table: list[tuple[str, str, str, str, str]] = [
        # variant, quantity, tolerance, scope, explanation
        ("ridge", "features_train", "forward", PARITY, ""),
        ("ridge", "features_test", "forward", PARITY, ""),
        ("ridge", "params", "exact", DEVIATION, ridge_params),
        ("ridge", "intercept", "forward", PARITY, ""),
        ("ridge", "predictions", "train", PARITY, ""),
        ("enet", "features_train", "forward", PARITY, ""),
        ("enet", "features_test", "forward", PARITY, ""),
        ("enet", "params", "exact", PARITY, ""),
        ("enet", "coef", "forward", PARITY, ""),
        ("enet", "intercept", "forward", PARITY, ""),
        ("enet", "predictions", "forward", PARITY, ""),
        ("enet_preset", "params", "exact", DEVIATION, solver),
        ("enet_preset", "coef", "statistical", DEVIATION, solver),
        ("enet_preset", "predictions", "statistical", DEVIATION, solver),
    ]
    return [
        StepSpec(
            name=f"{variant}/{quantity}",
            tolerance=tolerance,
            v1_key=f"{VARIANTS[variant].get('v1_variant', variant)}/{quantity}",
            scope=scope,
            description=f"{variant}: {quantity.replace('_', ' ')}",
            explanation=explanation,
        )
        for variant, quantity, tolerance, scope, explanation in table
    ]


def _effective_params(estimator: Any) -> np.ndarray:
    """Serialize the estimator type and parameters that influence the fit.

    ``random_state`` is dropped: it is inert for cyclic coordinate descent and for the dense
    ``Ridge`` solver, and 1.6.1 leaves it unset.

    :param estimator: Fitted sklearn estimator.
    :returns: One-element unicode array holding canonical JSON.
    """
    params = {k: v for k, v in estimator.get_params().items() if k != "random_state"}
    return np.array(json.dumps({"type": type(estimator).__name__, **params}, sort_keys=True, default=str))


def _v1_hyperparameters(setting: dict[str, Any]) -> dict[str, Any]:
    return {
        "alpha": setting["alpha"],
        "l1_ratio": setting["l1_ratio"],
        "cell_line_views": "gene_expression",
        "drug_views": "fingerprints",
        "gene_list": "landmark_genes",
    }


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the 1.6.1 ``ElasticNet`` model for every variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    from drevalpy.models.baselines.sklearn_models import ElasticNetModel

    from tools.v1_parity._v1_runner import tabular_inputs

    out: dict[str, np.ndarray] = {}
    for variant, setting in case.meta["variants"].items():
        if "v1_variant" in setting:
            continue
        data = tabular_inputs(case)
        model = ElasticNetModel()
        model.build_model(_v1_hyperparameters(setting))
        sink: dict[str, Any] = {}
        with (
            record_calls(model.model, "fit", sink, "features_train"),
            record_calls(model.model, "predict", sink, "features_test"),
        ):
            model.train(data.train, data.cell_line_input, data.drug_input)
            predictions = model.predict(
                data.test.cell_line_ids, data.test.drug_ids, data.cell_line_input, data.drug_input
            )
        out[f"{variant}/features_train"] = sink["features_train"]
        out[f"{variant}/features_test"] = sink["features_test"]
        out[f"{variant}/params"] = _effective_params(model.model)
        out[f"{variant}/coef"] = np.asarray(model.model.coef_)
        out[f"{variant}/intercept"] = np.atleast_1d(np.asarray(model.model.intercept_))
        out[f"{variant}/predictions"] = np.asarray(predictions)
    return out


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Train and predict with the v2 ``ElasticNet`` zoo preset for every variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    from drevalpy.models.drp_model import DRPModel
    from tools.v1_parity._v2_runner import capture_matrix_predictor, synthetic_gene_spec, tabular_inputs

    data = tabular_inputs(case)
    out: dict[str, np.ndarray] = {}
    for variant, setting in case.meta["variants"].items():
        hyperparameters = {
            "alpha": setting["alpha"],
            "l1_ratio": setting["l1_ratio"],
            **setting["v2_extra"],
        }
        model = DRPModel(synthetic_gene_spec(NAME), hyperparameters)
        sink: dict[str, Any] = {}
        with capture_matrix_predictor(sink):
            model.train(data.dataset, data.train)
            predictions = model.predict(data.dataset, data.test)
        out[f"{variant}/features_train"] = sink["features_train"]
        out[f"{variant}/features_test"] = sink["features_test"]
        out[f"{variant}/params"] = _effective_params(sink["estimator"])
        out[f"{variant}/coef"] = np.asarray(sink["estimator"].coef_)
        out[f"{variant}/intercept"] = np.atleast_1d(np.asarray(sink["estimator"].intercept_))
        out[f"{variant}/predictions"] = np.asarray(predictions)
    return out
