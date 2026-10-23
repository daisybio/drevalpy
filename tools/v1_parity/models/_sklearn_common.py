"""Shared implementation of the sklearn-baseline parity cases (no case module of its own).

A model module declares its 1.6.1 class, its v2 zoo preset and a dictionary of *variants*. A variant
holds the hyperparameters of both sides (``v1``, ``v2``; the string ``"$seed"`` stands for the case
seed), and optionally:

``scope``
    ``"parity"`` (default) or ``"deviation"``. A deviation variant only compares ``params`` and
    ``predictions``, recorded with their measured size; it needs an ``explanation``.
``v1_variant``
    Reuse the 1.6.1 run of another variant (one 1.6.1 run against several v2 settings).
``train_limit``
    Use only the first *n* training pairs on both sides.
``fingerprints``
    Name of the case array to use as drug features instead of ``fingerprints``.
``np_seed``
    Value of the global numpy seed set right before training on both sides (default: the case seed).
    1.6.1 leaves ``random_state`` unset for most estimators, which draws from the global numpy
    stream; seeding it and passing the same integer as ``random_state`` to v2 gives the identical
    stream (``RandomState(n)`` and ``np.random.seed(n)`` start from the same state).

Every parity variant also runs both estimators on the same float64 design matrix (``identical/*``),
which separates the algorithm from the float32 features of v2.

What is compared per variant: the design matrices that reach ``fit`` and ``predict`` (``features_*``),
the effective estimator parameters (``params``, ``random_state`` dropped and compared separately),
fitted attributes (``ATTRIBUTES``), neighbour sets (kNN) and the predictions.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, record_calls, tabular_case_arrays
from tools.v1_parity.compare import DEVIATION, PARITY, StepSpec

#: Fitted attributes read off the estimator when it has them.
ATTRIBUTES = (
    "coef_",
    "intercept_",
    "feature_importances_",
    "support_",
    "dual_coef_",
    "n_support_",
    "n_iter_",
    "estimator_weights_",
    "estimator_errors_",
    "train_score_",
    "validation_score_",
)
V1_VIEWS = {"cell_line_views": "gene_expression", "drug_views": "fingerprints", "gene_list": "landmark_genes"}


def build_sklearn_case(
    name: str,
    seed: int,
    variants: dict[str, dict[str, Any]],
    *,
    continuous_fingerprints: bool = False,
    **dimensions: int,
) -> Case:
    """Draw the shared tabular inputs and record the variants.

    :param name: Harness model name.
    :param seed: Seed of the input generator; also replaces ``"$seed"`` in the variants.
    :param variants: Variant settings, stored in the case metadata.
    :param continuous_fingerprints: Replace the binary fingerprints by continuous values (the binary ones stay
        available as ``fingerprints_binary``), so that distances do not tie.
    :param dimensions: ``n_samples``, ``n_drugs``, ``n_genes`` and ``n_bits`` of the inputs.
    :returns: The case.
    """
    arrays = tabular_case_arrays(seed, **dimensions)
    if continuous_fingerprints:
        arrays["fingerprints_binary"] = arrays["fingerprints"]
        rng = np.random.default_rng(seed + 7)
        arrays["fingerprints"] = rng.normal(size=arrays["fingerprints"].shape).astype(np.float32)
    meta = {
        "variants": json.loads(json.dumps(variants).replace('"$seed"', str(seed))),
        "n_train_pairs": int(len(arrays["train_pairs"])),
        "n_test_pairs": int(len(arrays["test_pairs"])),
        "gene_list": f"pre-selected: all {dimensions.get('n_genes', 40)} synthetic genes",
    }
    return Case(model=name, seed=seed, arrays=arrays, meta=meta)


def sklearn_steps(
    variants: dict[str, dict[str, Any]],
    *,
    attributes: dict[str, str],
    predictions: str = "forward",
    features: str = "forward",
    params: str = "exact",
    random_state: bool = True,
    random_state_note: str = "",
    identical: tuple[str, ...] | None = None,
) -> list[StepSpec]:
    """Declare the compared outputs of every variant.

    :param variants: Variant settings (their ``scope``, ``explanation`` and ``v1_variant`` keys are read).
    :param attributes: Fitted attributes to compare for parity variants, with their tolerance class.
    :param predictions: Tolerance class of the predictions of parity variants.
    :param features: Tolerance class of the design matrices.
    :param params: Tolerance class of the effective parameters (non-numeric: equal or not).
    :param random_state: Whether the estimator has a ``random_state`` to record as a deviation step.
    :param random_state_note: Explanation attached to the ``random_state`` step.
    :param identical: Fitted attributes compared (exact) in the ``identical`` run; defaults to the keys of
        *attributes*.
    :returns: The steps.
    """
    specs: list[StepSpec] = []
    exact_attributes = tuple(attributes) if identical is None else identical
    for variant, setting in variants.items():
        v1_variant = setting.get("v1_variant", variant)
        parity = setting.get("scope", PARITY) == PARITY
        if parity:
            table = [("features_train", features), ("features_test", features), ("params", params)]
            table += [(f"fitted_{a}", t) for a, t in attributes.items()] + [("predictions", predictions)]
        else:
            table = [("params", "exact"), ("predictions", "statistical")]
        specs += [
            StepSpec(
                name=f"{variant}/{quantity}",
                tolerance=tolerance,
                v1_key=f"{v1_variant}/{quantity}",
                scope=PARITY if parity else DEVIATION,
                description=f"{variant}: {quantity.replace('_', ' ')}",
                explanation="" if parity else setting["explanation"],
            )
            for quantity, tolerance in table
        ]
        if parity:
            same = [
                ("params", "exact"),
                *((f"fitted_{a}", "exact") for a in exact_attributes),
                ("predictions", "exact"),
            ]
            specs += [
                StepSpec(
                    name=f"{variant}/identical/{quantity}",
                    tolerance=tolerance,
                    v1_key=f"{v1_variant}/identical/{quantity}",
                    description=f"{variant}, identical float64 design matrix: {quantity.replace('_', ' ')}",
                )
                for quantity, tolerance in same
            ]
        if random_state and parity:
            specs.append(
                StepSpec(
                    name=f"{variant}/random_state",
                    tolerance="exact",
                    v1_key=f"{v1_variant}/random_state",
                    scope=DEVIATION,
                    description=f"{variant}: random_state of the estimator",
                    explanation=random_state_note,
                )
            )
    return specs


def effective_params(estimator: Any) -> np.ndarray:
    """Serialize the estimator type and the parameters that influence the fit, without ``random_state``.

    :param estimator: Fitted sklearn estimator.
    :returns: One-element unicode array holding canonical JSON.
    """
    params = {k: v for k, v in estimator.get_params().items() if "random_state" not in k}
    return np.array(json.dumps({"type": type(estimator).__name__, **params}, sort_keys=True, default=str))


def estimator_outputs(estimator: Any, x_test: np.ndarray) -> dict[str, np.ndarray]:
    """Read parameters, ``random_state``, fitted attributes and neighbour sets off a fitted estimator.

    :param estimator: Fitted sklearn estimator.
    :param x_test: Design matrix of the test pairs, to query the neighbours of a kNN estimator.
    :returns: ``params``, ``random_state`` (when the estimator has one), ``fitted_<attribute>`` and, for
        kNN, ``fitted_neighbors`` and ``fitted_distances`` of the test pairs.
    """
    out = {"params": effective_params(estimator)}
    if "random_state" in estimator.get_params(deep=False):
        out["random_state"] = np.array(str(estimator.random_state))
    for attribute in ATTRIBUTES:
        if hasattr(estimator, attribute):
            out[f"fitted_{attribute}"] = np.atleast_1d(np.asarray(getattr(estimator, attribute)))
    if hasattr(estimator, "kneighbors"):
        out["fitted_distances"], out["fitted_neighbors"] = estimator.kneighbors(x_test)
    return out


def design_matrices(case: Case, setting: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the 1.6.1 design matrices in float64: arcsinh, standardize on the training samples, append drugs.

    :param case: The shared case.
    :param setting: Variant settings (``train_limit`` and ``fingerprints`` are read).
    :returns: Training matrix, training response and test matrix.
    """
    from sklearn.preprocessing import StandardScaler

    expression = np.arcsinh(case.arrays["expression"].astype(np.float64))
    fingerprints = case.arrays[setting.get("fingerprints", "fingerprints")].astype(np.float64)
    pairs, test = case.arrays["train_pairs"][: setting.get("train_limit")], case.arrays["test_pairs"]
    scaler = StandardScaler().fit(expression[np.unique(pairs[:, 0])])

    def matrix(rows: np.ndarray) -> np.ndarray:
        return np.hstack([scaler.transform(expression[rows[:, 0]]), fingerprints[rows[:, 1]]])

    y = case.arrays["response"][pairs[:, 0], pairs[:, 1]].astype(np.float64)
    return matrix(pairs), y, matrix(test)


def identical_outputs(estimator: Any, case: Case, setting: dict[str, Any]) -> dict[str, np.ndarray]:
    """Fit an unfitted estimator on the float64 design matrix and read its outputs.

    :param estimator: Unfitted estimator of one side.
    :param case: The shared case.
    :param setting: Variant settings.
    :returns: ``identical/predictions`` plus the estimator outputs under ``identical/``.
    """
    x_train, y, x_test = design_matrices(case, setting)
    np.random.seed(setting.get("np_seed", case.seed))  # noqa: NPY002 - same global stream on both sides
    estimator.fit(x_train, y)
    out = {"predictions": np.asarray(estimator.predict(x_test), dtype=np.float64)}
    out.update(estimator_outputs(estimator, x_test))
    return {f"identical/{key}": value for key, value in out.items()}


def _variant_case(case: Case, setting: dict[str, Any]) -> Case:
    """Return *case* with the variant's drug features in the ``fingerprints`` slot.

    :param case: The shared case.
    :param setting: Variant settings.
    :returns: The same case, or a copy with the swapped array.
    """
    if "fingerprints" not in setting:
        return case
    arrays = {**case.arrays, "fingerprints": case.arrays[setting["fingerprints"]]}
    return Case(model=case.model, seed=case.seed, arrays=arrays, meta=case.meta)


def run_sklearn_v1(case: Case, module_path: str, class_name: str) -> dict[str, np.ndarray]:
    """Train and predict with a 1.6.1 sklearn model, once per variant.

    :param case: The shared case.
    :param module_path: Module of the 1.6.1 class.
    :param class_name: Name of the 1.6.1 class.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    from tools.v1_parity._v1_runner import tabular_inputs

    model_class = getattr(importlib.import_module(module_path), class_name)
    out: dict[str, np.ndarray] = {}
    for variant, setting in case.meta["variants"].items():
        if "v1_variant" in setting:
            continue
        data = tabular_inputs(_variant_case(case, setting))
        train = data.train
        if "train_limit" in setting:
            train.mask(np.arange(len(train)) < setting["train_limit"])
        model = model_class()
        model.build_model({**V1_VIEWS, **setting["v1"]})
        sink: dict[str, Any] = {}
        np.random.seed(setting.get("np_seed", case.seed))  # noqa: NPY002 - 1.6.1 draws from the legacy global state
        with (
            record_calls(model.model, "fit", sink, "features_train"),
            record_calls(model.model, "predict", sink, "features_test"),
        ):
            model.train(train, data.cell_line_input, data.drug_input)
            predictions = model.predict(
                data.test.cell_line_ids, data.test.drug_ids, data.cell_line_input, data.drug_input
            )
        out[f"{variant}/features_train"] = sink["features_train"]
        out[f"{variant}/features_test"] = sink["features_test"]
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        out.update({f"{variant}/{k}": v for k, v in estimator_outputs(model.model, sink["features_test"]).items()})
        if setting.get("scope", "parity") == "parity":
            fresh = model_class()
            fresh.build_model({**V1_VIEWS, **setting["v1"]})
            out.update({f"{variant}/{k}": v for k, v in identical_outputs(fresh.model, case, setting).items()})
    return out


def run_sklearn_v2(case: Case, preset: str) -> dict[str, np.ndarray]:
    """Train and predict with a v2 sklearn zoo preset, once per variant.

    :param case: The shared case.
    :param preset: Zoo preset name.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    from drevalpy.models.drp_model import DRPModel
    from drevalpy.types.data.split_mask import SplitMask
    from tools.v1_parity._v2_runner import capture_matrix_predictor, synthetic_gene_spec, tabular_inputs

    shape = (len(case.arrays["sample_ids"]), len(case.arrays["drug_ids"]))
    out: dict[str, np.ndarray] = {}
    for variant, setting in case.meta["variants"].items():
        data = tabular_inputs(_variant_case(case, setting))
        train = data.train
        if "train_limit" in setting:
            train = SplitMask.from_pairs(case.arrays["train_pairs"][: setting["train_limit"]], shape)
        model = DRPModel(synthetic_gene_spec(preset), dict(setting["v2"]))
        sink: dict[str, Any] = {}
        np.random.seed(setting.get("np_seed", case.seed))  # noqa: NPY002 - same global stream as the 1.6.1 side
        with capture_matrix_predictor(sink):
            model.train(data.dataset, train)
            predictions = model.predict(data.dataset, data.test)
        out[f"{variant}/features_train"] = sink["features_train"]
        out[f"{variant}/features_test"] = sink["features_test"]
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        out.update(
            {f"{variant}/{k}": v for k, v in estimator_outputs(sink["estimator"], sink["features_test"]).items()}
        )
        if setting.get("scope", "parity") == "parity":
            out.update(
                {f"{variant}/{k}": v for k, v in identical_outputs(sink["fresh_estimator"], case, setting).items()}
            )
    return out
