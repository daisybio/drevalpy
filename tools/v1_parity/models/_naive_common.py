"""Shared implementation of the naive-baseline parity cases (no case module of its own).

Every naive baseline is compared end to end: both sides are trained on the training pairs and
predict the training and the test pairs. The case is built so that the test pairs hit every
fallback of the group-mean models: a seen sample, an unseen sample of a seen tissue, an unseen
tissue (``Skin``), an unseen drug (``D05``) and an unseen (drug, tissue) combination.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, tabular_case_arrays
from tools.v1_parity.compare import PARITY, StepSpec

UNSEEN_DRUG = "D05"
UNSEEN_COMBINATION = ("D01", "Breast")
COMMON_FINDING = (
    "v2 evaluates the statistics in float64 (`ResponseBatch` casts the float32 response matrix to "
    "float64; the case values are float32-representable, so both sides average the identical numbers) "
    "with one-hot design products instead of dictionary lookups. Only the summation order differs, so "
    "every step is exact (rtol 1e-12). Fallbacks agree on all test pairs: unseen sample, unseen tissue, "
    "unseen drug and unseen (drug, tissue) combination all predict the dataset mean plus the effects that "
    "are known (see `coverage` in the case metadata)."
)
QUANTITIES = ("dataset_mean", "predictions_train", "predictions_test")


def _sorted_pairs(pairs: np.ndarray) -> np.ndarray:
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def build_naive_case(name: str, seed: int, variants: dict[str, dict[str, Any]]) -> Case:
    """Draw the tabular inputs and move pairs so that every fallback occurs in the test pairs.

    :param name: Harness model name.
    :param seed: Seed of the input generator.
    :param variants: Variant settings stored in the case metadata.
    :returns: The case.
    """
    arrays = tabular_case_arrays(seed, n_samples=36, n_drugs=6, n_genes=4, n_bits=4)
    tissues = arrays["tissues"].copy()
    tissues[-3:] = "Skin"
    arrays["tissues"] = tissues
    drug_ids = arrays["drug_ids"]
    train, test = arrays["train_pairs"], arrays["test_pairs"]
    moved = (drug_ids[train[:, 1]] == UNSEEN_DRUG) | (
        (drug_ids[train[:, 1]] == UNSEEN_COMBINATION[0]) & (tissues[train[:, 0]] == UNSEEN_COMBINATION[1])
    )
    arrays["train_pairs"] = train[~moved]
    arrays["test_pairs"] = _sorted_pairs(np.concatenate([test, train[moved]]))

    train_samples = set(arrays["sample_ids"][arrays["train_pairs"][:, 0]])
    train_drugs = set(drug_ids[arrays["train_pairs"][:, 1]])
    train_tissues = set(tissues[arrays["train_pairs"][:, 0]])
    combos = {(drug_ids[d], tissues[s]) for s, d in arrays["train_pairs"]}
    keys = [(arrays["sample_ids"][s], drug_ids[d], tissues[s]) for s, d in arrays["test_pairs"]]
    coverage = {
        "seen_pair_keys": sum(k[0] in train_samples and k[1] in train_drugs and (k[1], k[2]) in combos for k in keys),
        "unseen_sample": sum(k[0] not in train_samples for k in keys),
        "unseen_sample_seen_tissue": sum(k[0] not in train_samples and k[2] in train_tissues for k in keys),
        "unseen_tissue": sum(k[2] not in train_tissues for k in keys),
        "unseen_drug": sum(k[1] not in train_drugs for k in keys),
        "seen_drug_unseen_combination": sum(
            k[1] in train_drugs and k[2] in train_tissues and (k[1], k[2]) not in combos for k in keys
        ),
    }
    return Case(model=name, seed=seed, arrays=arrays, meta={"variants": variants, "coverage": coverage})


def naive_steps(variants: dict[str, dict[str, Any]], tolerance: str) -> list[StepSpec]:
    """Declare the compared outputs of every variant.

    :param variants: Variant names.
    :param tolerance: Tolerance class of every step.
    :returns: One parity step per variant and quantity.
    """
    return [
        StepSpec(
            name=f"{variant}/{quantity}",
            tolerance=tolerance,
            scope=PARITY,
            description=f"{variant}: {quantity.replace('_', ' ')}",
        )
        for variant in variants
        for quantity in QUANTITIES
    ]


def run_naive_v1(case: Case, v1_class: str) -> dict[str, np.ndarray]:
    """Train and predict with a 1.6.1 naive model, once per variant.

    :param case: The shared case.
    :param v1_class: Class name in ``drevalpy.models.baselines.naive_pred``.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    from drevalpy.datasets.dataset import FeatureDataset
    from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER
    from drevalpy.models.baselines import naive_pred

    from tools.v1_parity._v1_runner import tabular_inputs

    data = tabular_inputs(case)
    out: dict[str, np.ndarray] = {}
    for variant, setting in case.meta["variants"].items():
        cell_lines = FeatureDataset(
            features={
                sid: (
                    {CELL_LINE_IDENTIFIER: np.array([sid]), TISSUE_IDENTIFIER: np.array([tissue])}
                    if setting["tissue"]
                    else {CELL_LINE_IDENTIFIER: np.array([sid])}
                )
                for sid, tissue in zip(case.arrays["sample_ids"], case.arrays["tissues"], strict=True)
            }
        )
        drugs = FeatureDataset(features={d: {DRUG_IDENTIFIER: np.array([d])} for d in case.arrays["drug_ids"]})
        model = getattr(naive_pred, v1_class)()
        model.build_model({})
        model.train(data.train, cell_lines, drugs)
        out[f"{variant}/dataset_mean"] = np.atleast_1d(np.asarray(model.dataset_mean, dtype=np.float64))
        for key, pairs in (("train", data.train), ("test", data.test)):
            predictions = model.predict(pairs.cell_line_ids, pairs.drug_ids, cell_lines, drugs)
            out[f"{variant}/predictions_{key}"] = np.asarray(predictions, dtype=np.float64)
    return out


def run_naive_v2(case: Case, preset: str) -> dict[str, np.ndarray]:
    """Train and predict with a v2 naive zoo preset, once per variant.

    :param case: The shared case.
    :param preset: Zoo preset name.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    from drevalpy.models.drp_model import DRPModel
    from drevalpy.models.spec import ModelSpec
    from tools.v1_parity._v2_runner import tabular_inputs

    out: dict[str, np.ndarray] = {}
    for variant, setting in case.meta["variants"].items():
        data = tabular_inputs(case)
        obs = data.dataset.mdata.mod["response"].obs.copy()
        if not setting["tissue"]:
            obs["tissue"] = None  # annotations absent: the allow_missing path (a dropped column raises KeyError)
        data.dataset.mdata.obs = obs
        model = DRPModel(ModelSpec.from_name(preset))
        model.train(data.dataset, data.train)
        out[f"{variant}/dataset_mean"] = np.atleast_1d(np.float64(model._pipeline._predictor._dataset_mean))
        out[f"{variant}/predictions_train"] = np.asarray(model.predict(data.dataset, data.train), dtype=np.float64)
        out[f"{variant}/predictions_test"] = np.asarray(model.predict(data.dataset, data.test), dtype=np.float64)
    return out
