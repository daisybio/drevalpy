"""ElasticNet parity case: 1.6.1 ``ElasticNet`` against the v2 ``ElasticNet`` zoo preset.

The estimator input is compared, not the gene selection: both sides select ``landmark_genes`` from the full
view, but the synthetic gene names are in no shipped list, so both sides receive the same, already selected,
gene matrix (``synthetic_gene_spec`` drops the preset's list).

Variants (``ridge`` is the first set of the 1.6.1 ``hyperparameters.yaml`` grid, ``l1_ratio=0``,
``alpha=1``, which 1.6.1 builds as ``Ridge``):

``ridge``
    1.6.1 ``Ridge(alpha=1)`` against the v2 preset with no hyperparameter set (its defaults ``alpha=1``,
    ``l1_ratio=0`` build ``Ridge`` too).
``enet``
    1.6.1 ``ElasticNet(alpha=0.1, l1_ratio=0.5)`` (sklearn solver defaults) against the v2 preset with only
    ``alpha`` and ``l1_ratio`` set: the preset's solver settings are sklearn's defaults as well.

v2 holds features as float32 end to end while 1.6.1 computes in float64, so no class stricter
than ``forward`` holds for the features, and ``Ridge`` (solved in float32 by v2) needs ``train``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, record_calls, tabular_case_arrays
from tools.v1_parity.compare import PARITY, StepSpec

NAME = "ElasticNet"

VARIANTS: dict[str, dict[str, Any]] = {
    "ridge": {"l1_ratio": 0.0, "alpha": 1.0, "v2": {}},
    "enet": {"l1_ratio": 0.5, "alpha": 0.1, "v2": {"alpha": 0.1, "l1_ratio": 0.5}},
}

FINDINGS = [
    "1.6.1 selects the `landmark_genes` list while loading (hyperparameter `gene_list`; a missing gene "
    "raises); the v2 `ElasticNet` preset selects the same list with `scaledGeneExpression(gene_list=landmark_genes)`. "
    "The synthetic gene names are in no shipped list, so both sides get a pre-selected matrix and the selection "
    "itself is not compared here.",
    "v2 holds features as float32 end to end (`DenseViewFeaturizer` casts to float32, so the estimator "
    "receives a float32 matrix); 1.6.1 computes in float64. Features therefore agree to ~1e-7 relative "
    "(class forward), not exactly. Ridge solves in float32 in v2: refitting sklearn Ridge on the same "
    "matrix in float32 vs float64 differs by 2.4e-5 absolute (predictions ~7), which is the whole Ridge "
    "difference (class train).",
    "1.6.1 builds `sklearn.ElasticNet(alpha, l1_ratio)` with sklearn's solver defaults (max_iter=1000, "
    "tol=1e-4, cyclic, random_state unset). The v2 `elasticNet` predictor pins exactly these values "
    "(`non_tunable_hyperparameters`), so the effective parameters are equal and the coefficients agree to "
    "6e-8 (class forward) without any override.",
    "The first 1.6.1 hyperparameter set (l1_ratio=0, alpha=1) is `Ridge`; the v2 defaults are the same "
    "(alpha=1, l1_ratio=0) and build `Ridge(alpha)` without solver settings, as 1.6.1 does. The `ridge` variant "
    "runs the unmodified v2 preset.",
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

    :returns: Parity steps for ``ridge`` and ``enet``.
    """
    table: list[tuple[str, str, str]] = [
        # variant, quantity, tolerance
        ("ridge", "features_train", "forward"),
        ("ridge", "features_test", "forward"),
        ("ridge", "params", "exact"),
        ("ridge", "intercept", "forward"),
        ("ridge", "predictions", "train"),
        ("enet", "features_train", "forward"),
        ("enet", "features_test", "forward"),
        ("enet", "params", "exact"),
        ("enet", "coef", "forward"),
        ("enet", "intercept", "forward"),
        ("enet", "predictions", "forward"),
    ]
    return [
        StepSpec(
            name=f"{variant}/{quantity}",
            tolerance=tolerance,
            v1_key=f"{variant}/{quantity}",
            scope=PARITY,
            description=f"{variant}: {quantity.replace('_', ' ')}",
        )
        for variant, quantity, tolerance in table
    ]


def _effective_params(estimator: Any) -> np.ndarray:
    """Serialize the estimator type and all its parameters, ``random_state`` included.

    :param estimator: Fitted sklearn estimator.
    :returns: One-element unicode array holding canonical JSON.
    """
    params = estimator.get_params()
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
        model = DRPModel(synthetic_gene_spec(NAME), dict(setting["v2"]))
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
