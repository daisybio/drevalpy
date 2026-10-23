"""Shared implementation of the multi-omics tabular parity cases (no case module of its own).

The cases built on it (the MultiView forest and boosted trees, the proteomics sklearn variants) compare a
1.6.1 model with a v2 model stack on *real gene symbols*: every omics view is wider than the gene list the
1.6.1 loader selects from it, so the 1.6.1 selection (CSV loader, ``gene_list`` ordering) runs for real.

Inputs (:func:`build_omics_case`): ``view__<view>`` / ``view__<view>__names`` for the five views
(``gene_expression``, ``mutations``, ``copy_number_variation_gistic``, ``methylation``, ``proteomics``), the
gene lists as ``genelist__<name>``, ``fingerprints``, a ``fingerprints_constant`` (all-zero) twin and the
standard sample x drug pairs of :func:`tabular_case_arrays`.

A variant is a dictionary with

``v1``
    1.6.1 hyperparameters passed to ``build_model``.
``v1_class``
    Class name in ``V1_MODULE`` of the case module (variants of one case may differ).
``v2_config`` / ``v2_preset``
    A v2 model config in YAML layout (``sample_featurizer`` / ``drug_featurizer`` / ``predictor``), or the name
    of a zoo preset; ``v2`` holds the public hyperparameters. The v2 datasets are built per variant:
    ``preselect`` maps a view to a gene list that is applied to the v2 modality up front, for the featurizers
    that have no ``gene_list`` option (``raw``, ``pca``, ``normalizedProteomics``).
``fingerprints``
    Name of the case array used as drug features on both sides (default ``fingerprints``).
``drop_trailing``
    Number of trailing v2 design-matrix columns (drug features) that 1.6.1 does not have; dropped before the
    ``features_*`` steps.
``v1_scale_expression``
    Replace the leading expression block of the 1.6.1 design matrix by ``arcsinh`` + standard scaling fit on
    the unique training samples, which is what the v2 ``scaledGeneExpression`` featurizer does and the 1.6.1
    forest does not (it feeds the raw values).
``reference``
    ``[(view, gene_list | None), ...]`` of the float64 reference matrix for the ``identical/*`` run, and
    ``reference_fingerprints`` (``True`` appends the variant's drug features).
``scope`` / ``explanation`` / ``v1_variant``
    As in :mod:`tools.v1_parity.models._sklearn_common`.
``drug_spread``
    Record how much the predictions of a test sample vary across drugs.
``np_seed``
    As in :mod:`tools.v1_parity.models._sklearn_common`.
"""

from __future__ import annotations

import importlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, record_calls, tabular_case_arrays
from tools.v1_parity.compare import DEVIATION, PARITY, StepSpec
from tools.v1_parity.models._sklearn_common import effective_params, estimator_outputs, sklearn_steps

DATASET = "PARITY"
#: Per view: the gene list the 1.6.1 defaults select, how many extra genes the wide view holds, how its values are
#: drawn.
VIEWS: dict[str, dict[str, Any]] = {
    "gene_expression": {"list": "landmark_genes", "extra": 33, "kind": "expression"},
    "mutations": {"list": "drug_target_genes_all_drugs", "extra": 18, "kind": "binary"},
    "copy_number_variation_gistic": {"list": "drug_target_genes_all_drugs", "extra": 12, "kind": "cnv"},
    "methylation": {"list": "methylation_intersection", "extra": 6, "kind": "beta"},
    "proteomics": {"list": "drug_target_genes_all_drugs_proteomics", "extra": 70, "kind": "protein"},
}
GENE_LISTS = ("landmark_genes", "drug_target_genes_all_drugs", "methylation_intersection")
GENE_LISTS += ("drug_target_genes_all_drugs_proteomics",)
OMICS_ATTRIBUTES = ("feature_importances_",)


def _draw_view(kind: str, rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Draw one omics matrix (float32-representable)."""
    if kind == "expression":
        values = rng.gamma(shape=2.0, scale=30.0, size=shape)
    elif kind == "binary":
        values = (rng.random(shape) < 0.3).astype(np.float64)
    elif kind == "cnv":
        values = rng.choice([-2.0, -1.0, 0.0, 1.0, 2.0], size=shape, p=[0.05, 0.15, 0.6, 0.15, 0.05])
    elif kind == "beta":
        latent = rng.normal(size=(shape[0], 4))
        values = np.clip(0.5 + 0.15 * latent @ rng.normal(size=(4, shape[1])) + 0.05 * rng.normal(size=shape), 0, 1)
    else:
        values = np.exp(np.log(1e6) + rng.normal(scale=1.0, size=shape)) * np.exp(
            rng.normal(scale=0.2, size=(shape[0], 1))
        )
        values[rng.random(shape) < rng.uniform(0.0, 0.6, size=(1, shape[1]))] = np.nan
        values[rng.random(shape) < 0.01] = 0.0
    return values.astype(np.float32)


def build_omics_case(
    name: str,
    seed: int,
    variants: dict[str, dict[str, Any]],
    *,
    n_samples: int = 40,
    n_drugs: int = 8,
    n_bits: int = 32,
) -> Case:
    """Draw the multi-omics inputs and record the variants.

    :param name: Harness model name.
    :param seed: Seed of the input generator; also replaces ``"$seed"`` in the variants.
    :param variants: Variant settings, stored in the case metadata.
    :param n_samples: Number of samples (the last fifth never occurs in training).
    :param n_drugs: Number of drugs.
    :param n_bits: Width of the fingerprints.
    :returns: The case.
    """
    from drevalpy.components.featurizers.sample.gene_lists import resolve_gene_list

    arrays = tabular_case_arrays(seed, n_samples=n_samples, n_drugs=n_drugs, n_genes=1, n_bits=n_bits)
    rng = np.random.default_rng(seed + 101)
    lists = {list_name: np.array(resolve_gene_list(list_name)) for list_name in GENE_LISTS}
    views: dict[str, np.ndarray] = {}
    for view, spec in VIEWS.items():
        genes = np.array([*lists[spec["list"]], *(f"X{view[:4].upper()}{i:03d}" for i in range(spec["extra"]))])
        genes = genes[rng.permutation(len(genes))]
        views[view] = _draw_view(spec["kind"], rng, (n_samples, len(genes)))
        arrays[f"view__{view}"] = views[view]
        arrays[f"view__{view}__names"] = genes
    arrays.update({f"genelist__{k}": v for k, v in lists.items()})
    arrays["expression"], arrays["gene_names"] = arrays["view__gene_expression"], arrays["view__gene_expression__names"]

    def columns(view: str, list_name: str) -> np.ndarray:
        names = list(arrays[f"view__{view}__names"])
        return views[view][:, [names.index(g) for g in lists[list_name]]].astype(np.float64)

    signal = np.arcsinh(columns("gene_expression", "landmark_genes")) @ (
        rng.normal(size=len(lists["landmark_genes"])) / 30
    )
    signal += columns("mutations", "drug_target_genes_all_drugs") @ (
        rng.normal(size=len(lists["drug_target_genes_all_drugs"])) / 8
    )
    drug_effect = rng.normal(size=n_drugs)
    response = signal[:, None] * (1.0 + 0.3 * drug_effect[None, :]) + drug_effect[None, :]
    response += rng.normal(scale=0.1, size=response.shape)
    arrays["response"] = np.where(np.isnan(arrays["response"]), np.nan, response).astype(np.float32)
    arrays["fingerprints_zero"] = np.zeros_like(arrays["fingerprints"])
    meta = {
        "variants": json.loads(json.dumps(variants).replace('"$seed"', str(seed))),
        "n_train_pairs": int(len(arrays["train_pairs"])),
        "n_test_pairs": int(len(arrays["test_pairs"])),
        "views": {view: {"list": spec["list"], "n_columns": int(views[view].shape[1])} for view, spec in VIEWS.items()},
        "gene_lists": "shipped v2 lists written to the 1.6.1 loader's meta/gene_lists (1.6.1's CSVs are not offline)",
    }
    return Case(model=name, seed=seed, arrays=arrays, meta=meta)


def omics_steps(
    variants: dict[str, dict[str, Any]],
    *,
    attributes: dict[str, str],
    predictions: str,
    features: str,
    random_state_note: str = "",
    random_state_scope: str = DEVIATION,
    params: str = "exact",
    identical_attributes: str = "exact",
) -> list[StepSpec]:
    """Declare the compared outputs of every variant (see :func:`sklearn_steps`), plus recorded extras.

    :param variants: Variant settings.
    :param attributes: Fitted attributes of parity variants with their tolerance class.
    :param predictions: Tolerance class of the predictions of parity variants.
    :param features: Tolerance class of the design matrices of parity variants.
    :param random_state_note: Explanation attached to the ``random_state`` step when it is a deviation.
    :param random_state_scope: Scope of the ``random_state`` step: ``parity`` when both sides carry the same value,
        ``deviation`` when they differ (1.6.1 leaves it unset and draws from the global numpy stream).
    :param params: Tolerance class of the effective parameters.
    :param identical_attributes: Tolerance class of the fitted attributes in the ``identical`` run.
    :returns: The steps.
    """
    specs = sklearn_steps(
        variants,
        attributes=attributes,
        predictions=predictions,
        features=features,
        params=params,
        random_state_note=random_state_note,
        random_state_scope=random_state_scope,
    )
    only = {name for name, setting in variants.items() if setting.get("features_only")}

    def keep(spec: StepSpec) -> bool:
        variant = spec.name.split("/")[0]
        if variant not in only:
            return True
        return variants[variant].get("scope", PARITY) == PARITY and spec.name.split("/")[-1].startswith("features")

    specs = [s for s in specs if keep(s)]
    specs = [replace(s, tolerance=identical_attributes) if "/identical/fitted_" in s.name else s for s in specs]
    specs = [
        replace(s, tolerance=variants[s.name.split("/")[0]]["features"])
        if s.name.rsplit("/", 1)[-1] in ("features_train", "features_test")
        and "features" in variants[s.name.split("/")[0]]
        else s
        for s in specs
    ]
    for variant, setting in variants.items():
        v1_variant = setting.get("v1_variant", variant)
        parity = setting.get("scope", PARITY) == PARITY
        if variant in only or (not parity and not setting.get("scaled_features")):
            specs.append(
                StepSpec(
                    name=f"{variant}/n_columns",
                    tolerance="exact",
                    v1_key=f"{v1_variant}/n_columns",
                    scope=PARITY if parity else DEVIATION,
                    description=f"{variant}: number of design-matrix columns"
                    + (" (PCA component cap)" if parity else ""),
                    explanation="" if parity else setting["explanation"],
                )
            )
        if setting.get("scaled_features"):
            specs += [
                StepSpec(
                    name=f"{variant}/{quantity}",
                    tolerance=setting["scaled_features"],
                    v1_key=f"{v1_variant}/{quantity}_scaled",
                    description=f"{variant}: {quantity.replace('_', ' ')} (1.6.1 matrix after arcsinh + scaling)",
                )
                for quantity in ("features_train", "features_test")
            ]
        if setting.get("drug_spread"):
            specs.append(
                StepSpec(
                    name=f"{variant}/drug_spread",
                    tolerance="exact",
                    v1_key=f"{v1_variant}/drug_spread",
                    scope=DEVIATION,
                    description=f"{variant}: mean over test samples of the std of the prediction across drugs",
                    explanation=setting["spread_explanation"],
                )
            )
    return specs


def _selected(case: Case, view: str, list_name: str | None) -> np.ndarray:
    """Return the columns of a wide view in gene-list order (all columns when *list_name* is None)."""
    matrix = case.arrays[f"view__{view}"]
    if list_name is None:
        return matrix
    names = list(case.arrays[f"view__{view}__names"])
    return matrix[:, [names.index(g) for g in case.arrays[f"genelist__{list_name}"]]]


def reference_matrices(case: Case, setting: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the float64 reference design matrix both sides refit their estimator on (``identical/*``).

    :param case: The shared case.
    :param setting: Variant settings (``reference`` and ``reference_fingerprints``).
    :returns: Training matrix, training response and test matrix.
    """
    blocks = []
    for view, list_name in setting["reference"]:
        block = _selected(case, view, list_name).astype(np.float64)
        if view == "gene_expression":
            block = np.arcsinh(block)
        elif view == "proteomics":
            block = np.nan_to_num(np.log10(block), nan=0.0, posinf=0.0, neginf=0.0)
        blocks.append(block)
    fingerprints = case.arrays[setting.get("fingerprints", "fingerprints")].astype(np.float64)
    pairs, test = case.arrays["train_pairs"], case.arrays["test_pairs"]

    def matrix(rows: np.ndarray) -> np.ndarray:
        parts = [block[rows[:, 0]] for block in blocks]
        if setting.get("reference_fingerprints"):
            parts.append(fingerprints[rows[:, 1]])
        return np.hstack(parts)

    y = case.arrays["response"][pairs[:, 0], pairs[:, 1]].astype(np.float64)
    return matrix(pairs), y, matrix(test)


def _outputs(estimator: Any, x_test: np.ndarray) -> dict[str, np.ndarray]:
    """Read parameters, ``random_state`` and the fitted attributes; XGBoost parameters come from the booster."""
    out = estimator_outputs(estimator, x_test)
    if type(estimator).__module__.startswith("xgboost"):
        config = json.loads(estimator.get_booster().save_config())["learner"]
        keep = {
            "tree_train_param": config["gradient_booster"]["tree_train_param"],
            "learner_train_param": config["learner_train_param"],
            "base_score": config["learner_model_param"]["base_score"],
            "objective": config["objective"]["name"],
        }
        out["params"] = np.array(json.dumps({"type": type(estimator).__name__, **keep}, sort_keys=True))
    else:
        out["params"] = effective_params(estimator)
    return out


def _identical(estimator: Any, case: Case, setting: dict[str, Any]) -> dict[str, np.ndarray]:
    """Fit an unfitted estimator on the float64 reference matrix and read its outputs."""
    x_train, y, x_test = reference_matrices(case, setting)
    np.random.seed(setting.get("np_seed", case.seed))  # noqa: NPY002 - same global stream on both sides
    estimator.fit(x_train, y)
    out = {"predictions": np.asarray(estimator.predict(x_test), dtype=np.float64)}
    out.update(_outputs(estimator, x_test))
    return {f"identical/{key}": value for key, value in out.items()}


def write_v1_data(case: Case, root: Path) -> None:
    """Write the wide views and the gene lists as the CSVs the 1.6.1 loaders read.

    :param case: The shared case.
    :param root: Data directory; ``<root>/PARITY/<view>.csv`` and ``<root>/meta/gene_lists/<list>.csv``.
    """
    import pandas as pd

    (root / DATASET).mkdir(parents=True)
    (root / "meta" / "gene_lists").mkdir(parents=True)
    for view in VIEWS:
        frame = pd.DataFrame(
            case.arrays[f"view__{view}"].astype(np.float64),
            index=pd.Index(case.arrays["sample_ids"], name="cell_line_name"),
            columns=case.arrays[f"view__{view}__names"],
        )
        frame.to_csv(root / DATASET / f"{view}.csv")
    for list_name in GENE_LISTS:
        pd.DataFrame({"Symbol": case.arrays[f"genelist__{list_name}"]}).to_csv(
            root / "meta" / "gene_lists" / f"{list_name}.csv", index=False
        )


def _scale_expression(
    train: np.ndarray, test: np.ndarray, n_expression: int, unique_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return both design matrices with the leading expression block arcsinh-transformed and standardized.

    The scaler is fit on the *unique_rows* of the training matrix.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(np.arcsinh(train[unique_rows, :n_expression]))

    def apply(matrix: np.ndarray) -> np.ndarray:
        return np.hstack([scaler.transform(np.arcsinh(matrix[:, :n_expression])), matrix[:, n_expression:]])

    return apply(train), apply(test)


def run_omics_v1(case: Case, module_path: str, default_class: str) -> dict[str, np.ndarray]:
    """Train and predict with a 1.6.1 model through its own CSV loader, once per variant.

    :param case: The shared case.
    :param module_path: Module of the 1.6.1 classes.
    :param default_class: Class used by variants without ``v1_class``.
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    from tools.v1_parity._v1_runner import tabular_inputs

    module = importlib.import_module(module_path)
    out: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as tmp:
        write_v1_data(case, Path(tmp))
        for variant, setting in case.meta["variants"].items():
            if "v1_variant" in setting:
                continue
            model_class = getattr(module, setting.get("v1_class", default_class))
            data = tabular_inputs(_with_fingerprints(case, setting))
            model = model_class()
            model.build_model(dict(setting["v1"]))
            cell_line_input = model.load_cell_line_features(tmp, DATASET)
            sink: dict[str, Any] = {}
            np.random.seed(setting.get("np_seed", case.seed))  # noqa: NPY002 - 1.6.1 draws from the global state
            _pad_estimator(model.model, setting.get("v1_pad_columns", 0))
            with (
                record_calls(model.model, "fit", sink, "features_train"),
                record_calls(model.model, "predict", sink, "features_test"),
            ):
                model.train(data.train, cell_line_input, data.drug_input)
                predictions = model.predict(
                    data.test.cell_line_ids, data.test.drug_ids, cell_line_input, data.drug_input
                )
            _pad_estimator(model.model, setting.get("v1_pad_columns", 0))
            out[f"{variant}/drug_spread"] = _spread(model, data, cell_line_input)
            features_train, features_test = sink["features_train"], sink["features_test"]
            if setting.get("v1_scaled_features"):
                n_expression = _n_expression(model, cell_line_input)
                _, first = np.unique(data.train.cell_line_ids, return_index=True)
                scaled = _scale_expression(features_train, features_test, n_expression, first)
                out[f"{variant}/features_train_scaled"], out[f"{variant}/features_test_scaled"] = scaled
            out[f"{variant}/n_columns"] = np.array(features_train.shape[1])
            if setting.get("ignore_last"):
                features_train, features_test = features_train[:, :-1], features_test[:, :-1]
            out[f"{variant}/features_train"] = features_train
            out[f"{variant}/features_test"] = features_test
            out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
            out.update({f"{variant}/{k}": v for k, v in _outputs(model.model, sink["features_test"]).items()})
            if setting.get("scope", PARITY) == PARITY and not setting.get("features_only"):
                fresh = model_class()
                fresh.build_model(dict(setting["v1"]))
                out.update({f"{variant}/{k}": v for k, v in _identical(fresh.model, case, setting).items()})
    return out


def _with_fingerprints(case: Case, setting: dict[str, Any]) -> Case:
    """Return *case* with the variant's drug features in the ``fingerprints`` slot."""
    if "fingerprints" not in setting:
        return case
    arrays = {**case.arrays, "fingerprints": case.arrays[setting["fingerprints"]]}
    return Case(model=case.model, seed=case.seed, arrays=arrays, meta=case.meta)


def _pad_estimator(estimator: Any, n_columns: int) -> None:
    """Make an estimator see *n_columns* appended all-zero columns in ``fit`` and ``predict``.

    The instance attributes are removed again by :func:`record_calls`.
    """
    if not n_columns:
        return
    fit, predict = estimator.fit, estimator.predict

    def padded(x: np.ndarray) -> np.ndarray:
        return np.hstack([x, np.zeros((len(x), n_columns))])

    estimator.fit = lambda x, y: fit(padded(x), y)
    estimator.predict = lambda x: predict(padded(x))


def _n_expression(model: Any, cell_line_input: Any) -> int:
    """Return the width of the leading expression block of the 1.6.1 design matrix."""
    first = next(iter(cell_line_input.features.values()))
    return int(first["gene_expression"].shape[0]) if "gene_expression" in model.cell_line_views else 0


def _spread(model: Any, data: Any, cell_line_input: Any) -> np.ndarray:
    """Mean over test samples of the std of the 1.6.1 prediction across all drugs."""
    samples = np.unique(data.test.cell_line_ids)
    drugs = np.array(sorted(data.drug_input.identifiers))
    ids = np.repeat(samples, len(drugs))
    grid = model.predict(ids, np.tile(drugs, len(samples)), cell_line_input, data.drug_input)
    return np.array(np.asarray(grid, dtype=np.float64).reshape(len(samples), len(drugs)).std(axis=1).mean())


def _build_spec(case: Case, setting: dict[str, Any], name: str) -> Any:
    """Build the v2 model spec of a variant."""
    import yaml

    from drevalpy.models.config.io import from_dict
    from drevalpy.models.spec import ModelSpec

    if "v2_preset" in setting:
        return ModelSpec.from_name(setting["v2_preset"])
    return ModelSpec(name, from_dict(yaml.safe_load(json.dumps(setting["v2_config"]))))


def v2_dataset(case: Case, setting: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Build the v2 ``Dataset`` with every wide view; ``preselect`` views are restricted to their list.

    :param case: The shared case.
    :param setting: Variant settings.
    :returns: The dataset and the train and test masks.
    """
    import anndata as ad
    import mudata as md
    import pandas as pd

    from drevalpy.types.data.dataset import Dataset
    from drevalpy.types.data.split_mask import SplitMask

    sample_ids, drug_ids = case.arrays["sample_ids"], case.arrays["drug_ids"]
    response = ad.AnnData(
        X=case.arrays["response"].astype(np.float32),
        obs=pd.DataFrame({"cell_line_name": sample_ids, "tissue": case.arrays["tissues"]}, index=sample_ids),
        var=pd.DataFrame(index=drug_ids),
    )
    response.varm["morgan_fingerprint"] = case.arrays[setting.get("fingerprints", "fingerprints")].astype(np.float32)
    modalities = {"response": response}
    for view in VIEWS:
        list_name = setting.get("preselect", {}).get(view)
        names = case.arrays[f"view__{view}__names"]
        matrix = case.arrays[f"view__{view}"]
        if list_name is not None:
            names = case.arrays[f"genelist__{list_name}"]
            matrix = _selected(case, view, list_name)
        modalities[view] = ad.AnnData(
            X=matrix.astype(np.float32), obs=pd.DataFrame(index=sample_ids), var=pd.DataFrame(index=names)
        )
    shape = (len(sample_ids), len(drug_ids))
    return (
        Dataset(md.MuData(modalities), name="parity"),
        SplitMask.from_pairs(case.arrays["train_pairs"], shape),
        SplitMask.from_pairs(case.arrays["test_pairs"], shape),
    )


def run_omics_v2(case: Case, name: str) -> dict[str, np.ndarray]:
    """Train and predict with the v2 model of every variant.

    :param case: The shared case.
    :param name: Harness model name (names the recipe-built specs).
    :returns: Arrays keyed ``<variant>/<quantity>``.
    """
    from drevalpy.models.drp_model import DRPModel
    from drevalpy.types.data.split_mask import SplitMask
    from tools.v1_parity._v2_runner import capture_matrix_predictor

    shape = (len(case.arrays["sample_ids"]), len(case.arrays["drug_ids"]))
    out: dict[str, np.ndarray] = {}
    for variant, setting in case.meta["variants"].items():
        dataset, train, test = v2_dataset(case, setting)
        model = DRPModel(_build_spec(case, setting, name), dict(setting["v2"]))
        sink: dict[str, Any] = {}
        np.random.seed(setting.get("np_seed", case.seed))  # noqa: NPY002 - same global stream as the 1.6.1 side
        with capture_matrix_predictor(sink):
            model.train(dataset, train)
            predictions = model.predict(dataset, test)
        drop = setting.get("drop_trailing", 0)
        features = {key: np.array(sink[key], copy=True) for key in ("features_train", "features_test")}
        if setting.get("drug_spread"):
            n_samples, n_drugs = shape
            samples = np.unique(case.arrays["test_pairs"][:, 0])
            grid = np.array([(i, j) for i in samples for j in range(n_drugs)])
            spread = model.predict(dataset, SplitMask.from_pairs(grid, shape))
            spread = np.asarray(spread, dtype=np.float64).reshape(len(samples), n_drugs).std(axis=1).mean()
            out[f"{variant}/drug_spread"] = np.array(spread)
        out[f"{variant}/n_columns"] = np.array(sink["features_train"].shape[1] - drop)
        for key in ("features_train", "features_test"):
            out[f"{variant}/{key}"] = features[key][
                :, : features[key].shape[1] - drop - int(setting.get("ignore_last", False))
            ]
        out[f"{variant}/predictions"] = np.asarray(predictions, dtype=np.float64)
        out.update({f"{variant}/{k}": v for k, v in _outputs(sink["estimator"], features["features_test"]).items()})
        if setting.get("scope", PARITY) == PARITY and not setting.get("features_only"):
            out.update({f"{variant}/{k}": v for k, v in _identical(sink["fresh_estimator"], case, setting).items()})
    return out
