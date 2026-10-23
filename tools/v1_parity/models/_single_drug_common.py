"""Shared implementation of the single-drug sklearn parity cases (no case module of its own).

1.6.1 trains one model per drug: the training and test sets are masked to the drug and the
``SingleDrug*`` model receives no drug input. v2 trains one estimator per drug inside a single
``singleDrug*`` predictor, on features that were fitted once over all training pairs.

A variant holds the hyperparameters of both sides (``v1``, ``v2``; the string ``"$seed"`` stands for the
case seed) and optionally:

``pairs``
    ``"shared"`` uses a response matrix without gaps, so that every training sample was measured for every drug and
    the per-drug scaler of 1.6.1 and the global scaler of v2 see the same samples. The default
    (``"ragged"``) keeps the standard split, where each drug has its own subset of training samples.
``scope``, ``explanation``
    ``"deviation"`` variants record ``features_train`` and ``predictions`` with their measured size.
``np_seed``
    Value of the global numpy seed set right before the training of each drug in 1.6.1 (default: the
    case seed). v2 gets the same integer as ``random_state``, which starts the identical stream for
    every drug.

Every parity variant also fits both estimators of the first drug on the same float64 matrix
(``identical/*``), which separates the algorithm from the float32 features of v2.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest import mock

import numpy as np

from tools.v1_parity._bridge import Case, record_calls
from tools.v1_parity.compare import DEVIATION, PARITY, StepSpec
from tools.v1_parity.models._sklearn_common import V1_VIEWS, build_sklearn_case, estimator_outputs

N_DRUGS = 4
DRUGS = [f"D{i:02d}" for i in range(N_DRUGS)]


def build_single_drug_case(name: str, seed: int, variants: dict[str, dict[str, Any]]) -> Case:
    """Draw the shared inputs, including the response split in which every drug sees the same samples.

    :param name: Harness model name.
    :param seed: Seed of the input generator.
    :param variants: Variant settings, stored in the case metadata.
    :returns: The case; ``shared_response`` (gaps filled), ``shared_train_pairs`` and ``shared_test_pairs`` hold
        the alternative split.
    """
    case = build_sklearn_case(name, seed, variants, n_samples=40, n_drugs=N_DRUGS, n_genes=30, n_bits=8)
    arrays = case.arrays
    rng = np.random.default_rng(seed + 11)
    response = arrays["response"].astype(np.float64)
    column_mean = np.nanmean(response, axis=0)
    filled = np.where(np.isfinite(response), response, column_mean + rng.normal(scale=0.1, size=response.shape))
    arrays["shared_response"] = filled.astype(np.float32)
    last_train_sample = max(arrays["train_pairs"][:, 0])
    pairs = np.argwhere(np.isfinite(filled))
    is_train = pairs[:, 0] <= last_train_sample
    arrays["shared_train_pairs"] = pairs[is_train].astype(np.int64)
    arrays["shared_test_pairs"] = pairs[~is_train].astype(np.int64)
    return case


def variant_case(case: Case, setting: dict[str, Any]) -> Case:
    """Return *case* with the variant's pair split in the ``train_pairs`` / ``test_pairs`` slots.

    :param case: The shared case.
    :param setting: Variant settings.
    :returns: The same case, or a copy with the swapped arrays.
    """
    if setting.get("pairs") != "shared":
        return case
    arrays = {
        **case.arrays,
        "response": case.arrays["shared_response"],
        "train_pairs": case.arrays["shared_train_pairs"],
        "test_pairs": case.arrays["shared_test_pairs"],
    }
    return Case(model=case.model, seed=case.seed, arrays=arrays, meta=case.meta)


def single_drug_steps(
    variants: dict[str, dict[str, Any]],
    *,
    attributes: dict[str, str],
    predictions: str,
    random_state_note: str,
) -> list[StepSpec]:
    """Declare the compared outputs of every variant and drug.

    :param variants: Variant settings.
    :param attributes: Fitted attributes to compare for parity variants, with their tolerance class.
    :param predictions: Tolerance class of the end-to-end predictions of parity variants.
    :param random_state_note: Explanation attached to the ``random_state`` step.
    :returns: The steps.
    """
    specs: list[StepSpec] = []
    for variant, setting in variants.items():
        parity = setting.get("scope", PARITY) == PARITY
        scope = PARITY if parity else DEVIATION
        explanation = "" if parity else setting["explanation"]
        if parity:
            table = [("features_train", "forward"), ("features_test", "forward"), ("params", "exact")]
            table += [(f"fitted_{a}", t) for a, t in attributes.items()] + [("predictions", predictions)]
        else:
            table = [("features_train", "exact"), ("params", "exact"), ("predictions", "statistical")]
        specs += [
            StepSpec(
                name=f"{variant}/{drug}/{quantity}",
                tolerance=tolerance,
                scope=scope,
                description=f"{variant}, {drug}: {quantity.replace('_', ' ')}",
                explanation=explanation,
            )
            for drug in DRUGS
            for quantity, tolerance in table
        ]
        if not parity:
            continue
        same = [("params", "exact"), *((f"fitted_{a}", "exact") for a in attributes), ("predictions", "exact")]
        specs += [
            StepSpec(
                name=f"{variant}/identical/{quantity}",
                tolerance=tolerance,
                description=f"{variant}, identical float64 matrix of {DRUGS[0]}: {quantity.replace('_', ' ')}",
            )
            for quantity, tolerance in same
        ]
        specs.append(
            StepSpec(
                name=f"{variant}/{DRUGS[0]}/random_state",
                tolerance="exact",
                scope=DEVIATION,
                description=f"{variant}: random_state of the estimator",
                explanation=random_state_note,
            )
        )
    return specs


def _drug_matrices(case: Case, setting: dict[str, Any], drug_index: int) -> tuple[np.ndarray, ...]:
    """Build the float64 matrices of one drug, standardized on its own training samples.

    :param case: The shared case.
    :param setting: Variant settings (the ``pairs`` split is read).
    :param drug_index: Column of the response matrix.
    :returns: ``(x_train, y, x_test)``.
    """
    from sklearn.preprocessing import StandardScaler

    case = variant_case(case, setting)
    expression = np.arcsinh(case.arrays["expression"].astype(np.float64))
    pairs, test = case.arrays["train_pairs"], case.arrays["test_pairs"]
    train_rows, test_rows = pairs[pairs[:, 1] == drug_index, 0], test[test[:, 1] == drug_index, 0]
    scaler = StandardScaler().fit(expression[train_rows])
    y = case.arrays["response"][train_rows, drug_index].astype(np.float64)
    return scaler.transform(expression[train_rows]), y, scaler.transform(expression[test_rows])


def _fit_identical(estimator: Any, case: Case, setting: dict[str, Any]) -> dict[str, np.ndarray]:
    """Fit an unfitted estimator on the float64 matrix of the first drug and read its outputs.

    :param estimator: Unfitted estimator of one side.
    :param case: The shared case.
    :param setting: Variant settings.
    :returns: ``identical/predictions`` plus the estimator outputs under ``identical/``.
    """
    x_train, y, x_test = _drug_matrices(case, setting, 0)
    np.random.seed(setting.get("np_seed", case.seed))  # noqa: NPY002 - same global stream on both sides
    estimator.fit(x_train, y)
    out = {"predictions": np.asarray(estimator.predict(x_test), dtype=np.float64)}
    out.update(estimator_outputs(estimator, x_test))
    return {f"identical/{key}": value for key, value in out.items()}


def _drug_outputs(
    estimator: Any, x_fit: np.ndarray, x_query: np.ndarray, predictions: np.ndarray
) -> dict[str, np.ndarray]:
    """Collect what one drug's estimator received and learned.

    :param estimator: Fitted estimator of the drug.
    :param x_fit: Matrix passed to ``fit``.
    :param x_query: Matrix passed to ``predict``.
    :param predictions: Predictions of the model for the drug's test pairs.
    :returns: ``features_*``, the estimator outputs and ``predictions``.
    """
    out = {"features_train": x_fit, "features_test": x_query}
    out.update(estimator_outputs(estimator, x_query))
    out["predictions"] = np.asarray(predictions, dtype=np.float64)
    return out


def run_single_drug_v1(case: Case, module_path: str, class_name: str) -> dict[str, np.ndarray]:
    """Train and predict every drug with a 1.6.1 single-drug model, once per variant.

    :param case: The shared case.
    :param module_path: Module of the 1.6.1 class.
    :param class_name: Name of the 1.6.1 class.
    :returns: Arrays keyed ``<variant>/<drug>/<quantity>`` and ``<variant>/identical/<quantity>``.
    """
    import importlib

    from tools.v1_parity._v1_runner import tabular_inputs

    model_class = getattr(importlib.import_module(module_path), class_name)
    out: dict[str, np.ndarray] = {}
    for variant, setting in case.meta["variants"].items():
        current = variant_case(case, setting)
        data = tabular_inputs(current)
        hyperparameters = {**V1_VIEWS, **setting["v1"]}
        for index, drug in enumerate(case.arrays["drug_ids"]):
            train = data.train.copy()
            train.mask(train.drug_ids == drug)
            test = data.test.copy()
            test.mask(test.drug_ids == drug)
            model = model_class()
            model.build_model(hyperparameters)
            sink: dict[str, Any] = {}
            np.random.seed(setting.get("np_seed", case.seed))  # noqa: NPY002 - 1.6.1 draws from the global state
            with (
                record_calls(model.model, "fit", sink, "x_fit"),
                record_calls(model.model, "predict", sink, "x_query"),
            ):
                model.train(train, data.cell_line_input, None)
                predictions = model.predict(test.cell_line_ids, test.drug_ids, data.cell_line_input, None)
            result = _drug_outputs(model.model, sink["x_fit"], sink["x_query"], predictions)
            out.update({f"{variant}/{drug}/{key}": value for key, value in result.items()})
            if index == 0 and setting.get("scope", PARITY) == PARITY:
                fresh = model_class()
                fresh.build_model(hyperparameters)
                out.update({f"{variant}/{k}": v for k, v in _fit_identical(fresh.model, case, setting).items()})
    return out


@contextmanager
def _recording_estimators(
    predictor_class: Any, fits: dict[int, np.ndarray], queries: dict[int, np.ndarray]
) -> Iterator[None]:
    """Make every estimator created by *predictor_class* record the matrices passed to ``fit`` and ``predict``.

    :param predictor_class: v2 single-drug predictor class.
    :param fits: Receives the ``fit`` input, keyed by ``id`` of the estimator.
    :param queries: Receives the ``predict`` input, keyed by ``id`` of the estimator.
    :yields: Nothing; the patch is active inside the ``with`` block.
    """
    original: Callable[[Any], Any] = predictor_class._make_estimator

    def make(self: Any) -> Any:
        estimator = original(self)
        fit, predict = estimator.fit, estimator.predict

        def fit_recorder(x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
            fits[id(estimator)] = np.array(x, copy=True)
            return fit(x, y, *args, **kwargs)

        def predict_recorder(x: Any, *args: Any, **kwargs: Any) -> Any:
            queries[id(estimator)] = np.array(x, copy=True)
            return predict(x, *args, **kwargs)

        estimator.fit, estimator.predict = fit_recorder, predict_recorder
        return estimator

    with mock.patch.object(predictor_class, "_make_estimator", make):
        yield


def run_single_drug_v2(case: Case, preset: str, predictor_class: Any) -> dict[str, np.ndarray]:
    """Train and predict every drug with a v2 single-drug zoo preset, once per variant.

    :param case: The shared case.
    :param preset: Zoo preset name.
    :param predictor_class: The preset's predictor class, whose estimators are recorded.
    :returns: Arrays keyed ``<variant>/<drug>/<quantity>`` and ``<variant>/identical/<quantity>``.
    """
    from drevalpy.models.drp_model import DRPModel
    from tools.v1_parity._v2_runner import synthetic_gene_spec, tabular_inputs

    out: dict[str, np.ndarray] = {}
    for variant, setting in case.meta["variants"].items():
        current = variant_case(case, setting)
        data = tabular_inputs(current)
        test_drugs = current.arrays["test_pairs"][:, 1]
        fits: dict[int, np.ndarray] = {}
        queries: dict[int, np.ndarray] = {}
        np.random.seed(setting.get("np_seed", case.seed))  # noqa: NPY002 - random_state is set explicitly
        with _recording_estimators(predictor_class, fits, queries):
            model = DRPModel(synthetic_gene_spec(preset), dict(setting["v2"]))
            model.train(data.dataset, data.train)
            predictions = np.asarray(model.predict(data.dataset, data.test))
        estimators = model._pipeline._predictor._estimators
        for index, drug in enumerate(case.arrays["drug_ids"]):
            estimator = estimators[str(drug)]
            result = _drug_outputs(
                estimator, fits[id(estimator)], queries[id(estimator)], predictions[test_drugs == index]
            )
            out.update({f"{variant}/{drug}/{key}": value for key, value in result.items()})
        if setting.get("scope", PARITY) == PARITY:
            fresh = predictor_class(dict(setting["v2"]))._make_estimator()
            out.update({f"{variant}/{k}": v for k, v in _fit_identical(fresh, case, setting).items()})
    return out
