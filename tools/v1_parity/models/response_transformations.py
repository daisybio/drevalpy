"""Response-transformation parity case: 1.6.1 ``GroupMeanCenterer`` against the v2 one.

``drug_mean`` and ``drug_tissue_mean`` (and the ``standard`` scaler as a control) are fitted on the
training pairs, applied to the test pairs and inverted, on both sides:

* 1.6.1 through ``DrugResponseDataset.fit_transform`` / ``transform`` / ``inverse_transform``, which
  build the group labels from ``drug_ids`` and ``tissue``;
* v2 through ``group_labels``, ``fit_target_transform`` and ``apply_target_transform`` of
  ``models/_target_transform.py``, which the model pipeline calls.

The end-to-end steps run the naive mean predictor behind the transformation: 1.6.1 through
``drevalpy.experiment.train_and_predict``, v2 through ``DRPModel(NaivePredictor, response_transformation=...)``.
The mean of the training residuals is zero, so the predictions are the group offsets and test the
fallbacks end to end.

The case is built so that the test pairs contain every fallback of ``drug_tissue_mean``: a seen
(drug, tissue) combination, a seen drug in an unseen combination (``D01`` x ``Breast`` is held out of
training), a seen drug in an unseen tissue (``Skin``, which only the last test-only samples have), and
an unseen drug (``D05`` has no training pair).

``single_drug`` compares the 1.6.1 single-drug protocol, one model per drug fitted on that drug's pairs
only, with the v2 pipeline, which fits the transformation once and routes the drugs inside the predictor.
Both give the drug's own mean as the offset, so they must agree on every drug that has training pairs.
"""

from __future__ import annotations

import numpy as np

from tools.v1_parity._bridge import Case, tabular_case_arrays
from tools.v1_parity.compare import PARITY, StepSpec

NAME = "ResponseTransformations"

TRANSFORMATIONS = ("drug_mean", "drug_tissue_mean", "standard")
UNSEEN_DRUG = "D05"
UNSEEN_COMBINATION = ("D01", "Breast")

FINDINGS = [
    "`Dataset.get_tissue` reads the global `mdata.obs`, which the harness' v2 `tabular_inputs` leaves without a "
    "`tissue` column (only `response:tissue` after the MuData merge); real and synthetic datasets copy the response "
    "obs there (`mdata.obs = response.obs.copy()`). The case does the same itself.",
    "1.6.1 builds the group labels in `DrugResponseDataset._transformation_kwargs` from the `drug_ids` and "
    "`tissue` attributes of the dataset being transformed; v2 builds them in `group_labels` from the drug ids of "
    "the response batch and `Dataset.get_tissue(sample_ids)` of the dataset being fitted or predicted. For the same "
    "pairs the labels are identical, so the transformer is numerically equal (class exact).",
    "1.6.1 fits and applies the transformation after dropping pairs that lack features (`reduce_to`); v2 fits it "
    "on the observed pairs of the scope before the featurizers reject pairs (existing v2 behaviour for every "
    "transformer, see `tests/models/test_target_transform.py`). The case has features for every pair, so the two "
    "coincide here.",
    "1.6.1 trains a single-drug model per drug on that drug's pairs; v2 fits the transformation once over all "
    "drugs and routes the drugs inside the predictor. The per-drug offsets are the same, which `single_drug` "
    "checks. An unseen drug has no single-drug model in 1.6.1, so it is excluded from that step.",
    "`drevalpy run` fits `NaiveMeanEffectsPredictor` (the normalization reference) on the untransformed response "
    "in both versions; that exemption is in the run loop and is not part of this case.",
]


def _sorted_pairs(pairs: np.ndarray) -> np.ndarray:
    """Return *pairs* in row-major order.

    :param pairs: ``(n, 2)`` index array.
    :returns: The rows sorted by sample, then drug.
    """
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def build_case(seed: int) -> Case:
    """Draw the tabular inputs and move pairs so that every fallback occurs in the test pairs.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    arrays = tabular_case_arrays(seed, n_samples=36, n_drugs=6, n_genes=8, n_bits=8)
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

    train_drugs = set(drug_ids[arrays["train_pairs"][:, 1]])
    train_tissues = set(tissues[arrays["train_pairs"][:, 0]])
    combos = {(drug_ids[d], tissues[s]) for s, d in arrays["train_pairs"]}
    test_keys = [(drug_ids[d], tissues[s]) for s, d in arrays["test_pairs"]]
    coverage = {
        "seen_combination": sum(key in combos for key in test_keys),
        "seen_drug_unseen_combination": sum(
            key not in combos and key[0] in train_drugs and key[1] in train_tissues for key in test_keys
        ),
        "seen_drug_unseen_tissue": sum(key[0] in train_drugs and key[1] not in train_tissues for key in test_keys),
        "unseen_drug": sum(key[0] not in train_drugs for key in test_keys),
    }
    return Case(
        model=NAME, seed=seed, arrays=arrays, meta={"transformations": list(TRANSFORMATIONS), "coverage": coverage}
    )


def steps() -> list[StepSpec]:
    """Declare the compared outputs; every one is a parity step of class ``exact``.

    :returns: One step per transformation and quantity, plus the single-drug protocol.
    """
    quantities = {
        "train_residuals": "residuals of the training pairs after the fit",
        "test_residuals": "residuals of the test pairs, with unseen drugs, tissues and combinations",
        "test_roundtrip": "inverse transform of the test residuals (the original test responses)",
        "predictions": "naive mean predictor trained on the residuals, inverse transformed",
    }
    table = [
        StepSpec(name=f"{name}/{quantity}", tolerance="exact", scope=PARITY, description=f"{name}: {text}")
        for name in TRANSFORMATIONS
        for quantity, text in quantities.items()
    ]
    table.append(
        StepSpec(
            name="single_drug/predictions",
            tolerance="exact",
            v1_key="single_drug/predictions",
            v2_key="single_drug/predictions",
            description="drug_mean with one model per drug (1.6.1) against the routed v2 pipeline, seen drugs only",
        )
    )
    return table


def _groups_pairs(case: Case, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sample ids, drug ids, tissues and float64 responses of the pairs under *key*.

    :param case: The case.
    :param key: ``"train_pairs"`` or ``"test_pairs"``.
    :returns: The four aligned arrays.
    """
    pairs = case.arrays[key]
    samples, drugs = case.arrays["sample_ids"][pairs[:, 0]], case.arrays["drug_ids"][pairs[:, 1]]
    tissues = case.arrays["tissues"][pairs[:, 0]]
    return samples, drugs, tissues, case.arrays["response"][pairs[:, 0], pairs[:, 1]].astype(np.float64)


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Fit, apply and invert every transformation with the 1.6.1 dataset API, and run the naive mean behind it.

    :param case: The shared case.
    :returns: Arrays keyed ``<transformation>/<quantity>`` and ``single_drug/predictions``.
    """
    from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

    from drevalpy.experiment import train_and_predict
    from drevalpy.models import MODEL_FACTORY
    from drevalpy.utils import get_response_transformation

    def dataset(key: str, only_drug: str | None = None) -> DrugResponseDataset:
        samples, drugs, tissues, response = _groups_pairs(case, key)
        keep = np.ones(len(drugs), dtype=bool) if only_drug is None else drugs == only_drug
        return DrugResponseDataset(
            response=response[keep],
            cell_line_ids=samples[keep],
            drug_ids=drugs[keep],
            tissues=tissues[keep],
            dataset_name="parity",
        )

    cell_lines = FeatureDataset(features={s: {"cell_line_id": np.array([s])} for s in case.arrays["sample_ids"]})
    drugs = FeatureDataset(features={d: {"drug_id": np.array([d])} for d in case.arrays["drug_ids"]})

    def end_to_end(name: str, train: DrugResponseDataset, test: DrugResponseDataset) -> np.ndarray:
        model = MODEL_FACTORY["NaivePredictor"]()
        predicted = train_and_predict(
            model=model,
            hpams={},
            path_data="unused",
            train_dataset=train,
            prediction_dataset=test,
            response_transformation=get_response_transformation(name),
            cl_features=cell_lines,
            drug_features=drugs,
        )
        return np.asarray(predicted.predictions)

    out: dict[str, np.ndarray] = {}
    for name in TRANSFORMATIONS:
        transformation = get_response_transformation(name)
        train, test = dataset("train_pairs"), dataset("test_pairs")
        train.fit_transform(transformation)
        test.transform(transformation)
        out[f"{name}/train_residuals"] = train.response.copy()
        out[f"{name}/test_residuals"] = test.response.copy()
        test.inverse_transform(transformation)
        out[f"{name}/test_roundtrip"] = test.response.copy()
        out[f"{name}/predictions"] = end_to_end(name, dataset("train_pairs"), dataset("test_pairs"))

    test_drugs = dataset("test_pairs").drug_ids
    single = np.full(len(test_drugs), np.nan)
    for drug in np.unique(dataset("train_pairs").drug_ids):
        single[test_drugs == drug] = end_to_end("drug_mean", dataset("train_pairs", drug), dataset("test_pairs", drug))
    out["single_drug/predictions"] = single
    return out


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Fit, apply and invert every transformation with the v2 pipeline helpers, and run the naive mean behind it.

    :param case: The shared case.
    :returns: Arrays keyed like :func:`run_v1`.
    """
    from drevalpy.models._target_transform import apply_target_transform, fit_target_transform, group_labels
    from drevalpy.models.drp_model import DRPModel
    from drevalpy.models.spec import ModelSpec
    from drevalpy.types.data.batch.response_batch import ResponseBatch
    from drevalpy.utils.response_transform import get_response_transformation
    from tools.v1_parity._v2_runner import tabular_inputs

    data = tabular_inputs(case)
    data.dataset.mdata.obs = data.dataset.mdata.mod["response"].obs.copy()
    out: dict[str, np.ndarray] = {}
    for name in TRANSFORMATIONS:
        prototype = get_response_transformation(name)
        train = ResponseBatch.from_dataset(data.dataset, data.train, observed_only=True)
        test = ResponseBatch.from_dataset(data.dataset, data.test, observed_only=True)
        fitted = fit_target_transform(
            prototype, train.response.astype(np.float64), group_labels(prototype, data.dataset, train)
        )
        test_groups = group_labels(fitted, data.dataset, test)
        test_values = test.response.astype(np.float64)
        residuals = apply_target_transform(fitted, test_values, groups=test_groups)
        out[f"{name}/train_residuals"] = apply_target_transform(
            fitted, train.response.astype(np.float64), groups=group_labels(fitted, data.dataset, train)
        )
        out[f"{name}/test_residuals"] = residuals
        out[f"{name}/test_roundtrip"] = apply_target_transform(fitted, residuals, groups=test_groups, inverse=True)
        model = DRPModel(
            ModelSpec.from_name("NaivePredictor"), response_transformation=get_response_transformation(name)
        )
        model.train(data.dataset, data.train)
        out[f"{name}/predictions"] = np.asarray(model.predict(data.dataset, data.test), dtype=np.float64)

    single = out["drug_mean/predictions"].copy()
    test_drugs = case.arrays["drug_ids"][case.arrays["test_pairs"][:, 1]]
    train_drugs = set(case.arrays["drug_ids"][case.arrays["train_pairs"][:, 1]])
    single[~np.isin(test_drugs, list(train_drugs))] = np.nan
    out["single_drug/predictions"] = single
    return out
