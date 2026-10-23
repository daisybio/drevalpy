"""SingleDrugKNN parity case: 1.6.1 ``SingleDrugKNN`` against the v2 ``SingleDrugKNN`` zoo preset.

The case holds the 867 shipped ``landmark_genes`` in shuffled order plus extra genes, so the v2
``landmarkGenes`` selection is exercised. 1.6.1 selects the list while *loading* CSVs (the in-memory
harness skips that), so the 1.6.1 side receives the matrix reordered to the list by the rule of
``load_and_select_gene_features`` (list order, all list genes required). The response of every drug
depends on three drug-specific genes, so the mRMR selection is not arbitrary.

Variants (hyperparameters of both sides in ``VARIANTS``):

``default``
    The 1.6.1 defaults: ``n_neighbors=3``, ``weights=distance``, ``mrmr_k=20``.
``uniform``
    ``n_neighbors=10``, ``weights=uniform``, ``mrmr_k=8``.
``no_mrmr``
    ``mrmr_k=0``: no selection, the kNN runs on every landmark gene.
``clamped``
    ``n_neighbors=50``, more than a drug has training pairs: both sides reduce it to the row count.

For each drug the end-to-end steps compare what enters the per-drug pipeline (the arcsinh-ed,
standardized matrix), the mRMR selection, the neighbours used, the neighbour sets and distances of
the test pairs, and the predictions. ``<variant>/identical/*`` runs the estimator of each side on the
same float64 matrix of the first drug, isolating the algorithm from the float32 features of v2.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case, tabular_case_arrays
from tools.v1_parity.compare import PARITY, StepSpec

NAME = "SingleDrugKNN"

VARIANTS: dict[str, dict[str, Any]] = {
    "default": {"n_neighbors": 3, "weights": "distance", "mrmr_k": 20},
    "uniform": {"n_neighbors": 10, "weights": "uniform", "mrmr_k": 8},
    "no_mrmr": {"n_neighbors": 5, "weights": "distance", "mrmr_k": 0},
    "clamped": {"n_neighbors": 50, "weights": "uniform", "mrmr_k": 20},
}

N_EXTRA_GENES = 25
QUANTITIES = ("scaled_train", "scaled_test", "selected", "n_neighbors", "neighbors", "distances", "predictions")

FINDINGS = [
    "1.6.1 selects `landmark_genes` while loading the CSVs and fits one `StandardScaler` on the unique "
    "training cell lines of the drug, then runs `Pipeline(mrmr, knn)`. v2 selects the list in the "
    "`landmarkGenes` featurizer (standardize=false, arcsinh) and runs `Pipeline(StandardScaler, "
    "MRMRSelector, NeighborCappedKNN)` per drug, so the scaler is fitted on the training pairs of the "
    "drug, the same rows (one pair per sample and drug).",
    "v2 holds features as float32 end to end, 1.6.1 as float64. The end-to-end feature, distance and "
    "prediction steps therefore agree to float32 precision (class forward), not exactly; the mRMR "
    "selection and the neighbour sets are compared exactly. The `identical` steps feed both estimators "
    "the same float64 matrix and are exact.",
    "The 1.6.1 default (first grid point) is n_neighbors=3, v2's `SingleDrugKNN` preset declares the same "
    "default and extends the tunable range to 20 (1.6.1 grid: 3, 5, 10, 20).",
    "v2 clamps `n_neighbors` inside the estimator when it is fitted (`NeighborCappedKNN`); 1.6.1 clamps in "
    "`train` through `set_params`. Both give `min(n_neighbors, n_rows)`.",
]


def build_case(seed: int) -> Case:
    """Draw the shared inputs: landmark genes in shuffled order plus extras, drug-specific signal.

    :param seed: Seed of the input generator.
    :returns: The case.
    """
    from drevalpy.components.featurizers.sample.gene_lists import resolve_gene_list

    landmark = np.array(resolve_gene_list("landmark_genes"))
    rng = np.random.default_rng(seed + 1)
    names = np.concatenate([landmark, np.array([f"XTRA{i:03d}" for i in range(N_EXTRA_GENES)])])
    names = names[rng.permutation(len(names))]
    arrays = tabular_case_arrays(seed, n_samples=40, n_drugs=4, n_genes=len(names), n_bits=8)
    arrays["gene_names"] = names
    position = {gene: i for i, gene in enumerate(names)}
    signal = np.arcsinh(arrays["expression"].astype(np.float64))
    response = np.full(arrays["response"].shape, np.nan)
    observed = np.isfinite(arrays["response"])
    for j in range(response.shape[1]):
        drivers = rng.choice(landmark, size=3, replace=False)
        weights = rng.normal(size=3)
        value = sum(w * signal[:, position[g]] for w, g in zip(weights, drivers, strict=True))
        response[:, j] = value + rng.normal(scale=0.2, size=len(value))
    arrays["response"] = np.where(observed, response, np.nan).astype(np.float32)
    arrays["gene_list"] = landmark
    return Case(model=NAME, seed=seed, arrays=arrays, meta={"variants": VARIANTS, "gene_list": "landmark_genes"})


def steps() -> list[StepSpec]:
    """Declare the compared outputs, per variant and drug.

    :returns: Parity steps; selection, neighbour sets and the ``identical`` runs are exact.
    """
    drugs = [f"D{i:02d}" for i in range(4)]
    tolerance = {
        "scaled_train": "forward",
        "scaled_test": "forward",
        "selected": "exact",
        "n_neighbors": "exact",
        "neighbors": "exact",
        "distances": "forward",
        "predictions": "forward",
    }
    specs = [
        StepSpec(
            name=f"{variant}/{drug}/{quantity}",
            tolerance=tolerance[quantity],
            scope=PARITY,
            description=f"{variant}, {drug}: {quantity.replace('_', ' ')}",
        )
        for variant in VARIANTS
        for drug in drugs
        for quantity in QUANTITIES
    ]
    specs += [
        StepSpec(
            name=f"{variant}/identical/{quantity}",
            tolerance="exact",
            scope=PARITY,
            description=f"{variant}, identical float64 input to both estimators: {quantity}",
        )
        for variant in VARIANTS
        for quantity in ("selected", "n_neighbors", "neighbors", "distances", "predictions")
    ]
    return specs


def _pipeline_outputs(pipeline: Any, x_fit: np.ndarray, x_query: np.ndarray) -> dict[str, np.ndarray]:
    """Read the selection, neighbour sets and predictions off a fitted ``[(scale), mrmr, knn]`` pipeline.

    :param pipeline: Fitted sklearn pipeline whose last step is the kNN regressor.
    :param x_fit: Matrix the pipeline was fitted on.
    :param x_query: Matrix of the test rows.
    :returns: The matrices as they enter the selection (standardized), the selection, ``n_neighbors``,
        neighbour indices, distances and the pipeline's predictions.
    """
    scaler = pipeline.named_steps.get("scale")
    mrmr = pipeline.named_steps["mrmr"]
    knn = pipeline.named_steps["knn"]
    selected = mrmr.selected_ if hasattr(mrmr, "selected_") else np.arange(x_fit.shape[1])
    distances, neighbors = knn.kneighbors(pipeline[:-1].transform(x_query))
    return {
        "scaled_train": scaler.transform(x_fit) if scaler is not None else x_fit,
        "scaled_test": scaler.transform(x_query) if scaler is not None else x_query,
        "selected": np.asarray(selected),
        "n_neighbors": np.array(knn.n_neighbors),
        "neighbors": neighbors,
        "distances": distances,
        "predictions": np.asarray(pipeline.predict(x_query), dtype=np.float64),
    }


def _drug_data(case: Case, drug_index: int, columns: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the arcsinh-ed float64 landmark matrices and responses of one drug.

    :param case: The shared case.
    :param drug_index: Column of the response matrix.
    :param columns: Column positions of the list genes, in list order.
    :returns: ``(train_x, train_y, test_x)``.
    """
    expression = np.arcsinh(case.arrays["expression"].astype(np.float64))[:, columns]
    pairs = case.arrays["train_pairs"]
    train_rows = pairs[pairs[:, 1] == drug_index, 0]
    test_pairs = case.arrays["test_pairs"]
    test_rows = test_pairs[test_pairs[:, 1] == drug_index, 0]
    y = case.arrays["response"][train_rows, drug_index].astype(np.float64)
    return expression[train_rows], y, expression[test_rows]


def _identical_outputs(case: Case, estimator: Any, scale_outside: bool) -> dict[str, np.ndarray]:
    """Fit *estimator* on the float64 matrix of the first drug and read its outputs.

    :param case: The shared case.
    :param estimator: Unfitted ``Pipeline(mrmr, knn)`` (1.6.1) or ``Pipeline(scale, mrmr, knn)`` (v2).
    :param scale_outside: Whether to standardize before the pipeline, as 1.6.1 does.
    :returns: Selection, ``n_neighbors``, neighbours, distances and predictions.
    """
    from sklearn.preprocessing import StandardScaler

    names = list(case.arrays["gene_names"])
    columns = np.array([names.index(g) for g in case.arrays["gene_list"]])
    x_train, y, x_test = _drug_data(case, 0, columns)
    if scale_outside:
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    estimator.fit(x_train, y)
    out = _pipeline_outputs(estimator, x_train, x_test)
    return {key: out[key] for key in ("selected", "n_neighbors", "neighbors", "distances", "predictions")}


@contextmanager
def _record_pipeline_inputs(
    pipeline_class: Any, fits: dict[int, np.ndarray], queries: dict[int, np.ndarray]
) -> Iterator[None]:
    """Record, per pipeline instance, the matrices passed to ``fit`` and ``predict``.

    :param pipeline_class: ``sklearn.pipeline.Pipeline``.
    :param fits: Receives the ``fit`` input, keyed by ``id`` of the pipeline.
    :param queries: Receives the ``predict`` input, keyed by ``id`` of the pipeline.
    :yields: Nothing; the patches are active inside the ``with`` block.
    """
    original_fit, original_predict = pipeline_class.fit, pipeline_class.predict

    def fit_recorder(self: Any, X: Any, y: Any = None, **params: Any) -> Any:  # noqa: N803
        fits[id(self)] = np.array(X, copy=True)
        return original_fit(self, X, y, **params)

    def predict_recorder(self: Any, X: Any, **params: Any) -> Any:  # noqa: N803
        queries[id(self)] = np.array(X, copy=True)
        return original_predict(self, X, **params)

    pipeline_class.fit, pipeline_class.predict = fit_recorder, predict_recorder
    try:
        yield
    finally:
        pipeline_class.fit, pipeline_class.predict = original_fit, original_predict


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Train and predict every drug with the 1.6.1 ``SingleDrugKNN`` model, per variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<drug>/<quantity>`` and ``<variant>/identical/<quantity>``.
    """
    from drevalpy.models.baselines.singledrug_baselines import SingleDrugKNN

    from tools.v1_parity._bridge import record_calls
    from tools.v1_parity._v1_runner import tabular_inputs

    data = tabular_inputs(case)
    # 1.6.1 reorders to the list while loading (load_and_select_gene_features): all list genes, list order.
    names = list(case.arrays["gene_names"])
    columns = [names.index(g) for g in case.arrays["gene_list"]]
    for views in data.cell_line_input.features.values():
        views["gene_expression"] = views["gene_expression"][columns]
    data.cell_line_input.meta_info["gene_expression"] = np.array(case.arrays["gene_list"])

    out: dict[str, np.ndarray] = {}
    for variant, setting in case.meta["variants"].items():
        hyperparameters = {
            **setting,
            "cell_line_views": "gene_expression",
            "gene_list": "landmark_genes",
        }
        for drug in case.arrays["drug_ids"]:
            train = data.train.copy()
            train.mask(train.drug_ids == drug)
            test = data.test.copy()
            test.mask(test.drug_ids == drug)
            model = SingleDrugKNN()
            model.build_model(hyperparameters)
            sink: dict[str, Any] = {}
            with (
                record_calls(model.model, "fit", sink, "x_fit"),
                record_calls(model.model, "predict", sink, "x_query"),
            ):
                model.train(train, data.cell_line_input, None)
                predictions = model.predict(test.cell_line_ids, test.drug_ids, data.cell_line_input, None)
            result = _pipeline_outputs(model.model, sink["x_fit"], sink["x_query"])
            result["predictions"] = np.asarray(predictions, dtype=np.float64)
            out.update({f"{variant}/{drug}/{key}": value for key, value in result.items()})

        model = SingleDrugKNN()
        model.build_model(hyperparameters)
        n_rows = int(np.count_nonzero(case.arrays["train_pairs"][:, 1] == 0))
        if n_rows < setting["n_neighbors"]:  # what SingleDrugKNN.train does
            model.model.set_params(knn__n_neighbors=n_rows)
        identical = _identical_outputs(case, model.model, scale_outside=True)
        out.update({f"{variant}/identical/{key}": value for key, value in identical.items()})
    return out


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Train and predict every drug with the v2 ``SingleDrugKNN`` zoo preset, per variant.

    :param case: The shared case.
    :returns: Arrays keyed ``<variant>/<drug>/<quantity>`` and ``<variant>/identical/<quantity>``.
    """
    from sklearn.pipeline import Pipeline

    from drevalpy.components.predictors.sklearn_models import SingleDrugKNNPredictor
    from drevalpy.models.drp_model import DRPModel
    from drevalpy.models.spec import ModelSpec
    from tools.v1_parity._v2_runner import tabular_inputs

    data = tabular_inputs(case)
    test_drugs = case.arrays["test_pairs"][:, 1]
    out: dict[str, np.ndarray] = {}
    for variant, setting in case.meta["variants"].items():
        fits: dict[int, np.ndarray] = {}
        queries: dict[int, np.ndarray] = {}
        with _record_pipeline_inputs(Pipeline, fits, queries):
            model = DRPModel(ModelSpec.from_name(NAME), dict(setting))
            model.train(data.dataset, data.train)
            predictions = model.predict(data.dataset, data.test)
        estimators = model._pipeline._predictor._estimators
        for index, drug in enumerate(case.arrays["drug_ids"]):
            pipeline = estimators[str(drug)]
            result = _pipeline_outputs(pipeline, fits[id(pipeline)], queries[id(pipeline)])
            result["predictions"] = np.asarray(predictions)[test_drugs == index]
            out.update({f"{variant}/{drug}/{key}": value for key, value in result.items()})

        estimator = SingleDrugKNNPredictor(dict(setting))._make_estimator()
        identical = _identical_outputs(case, estimator, scale_outside=False)
        out.update({f"{variant}/identical/{key}": value for key, value in identical.items()})
    return out
