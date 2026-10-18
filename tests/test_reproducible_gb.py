"""Tests for the deterministic GradientBoosting variant."""

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import MODEL_FACTORY
from drevalpy.models.baselines.reproducible_gb import GradientBoostingReproducible
from drevalpy.models.baselines.sklearn_models import GradientBoosting

_HPAMS = {
    "cell_line_views": "gene_expression",
    "drug_views": "fingerprints",
    "gene_list": None,
    "max_iter": 20,
    "learning_rate": 0.1,
    "max_depth": 3,
    "early_stopping": True,
    "random_state": 42,
}


def _toy_data(n_cells: int = 24, n_drugs: int = 5, genes: int = 6, bits: int = 8):
    """
    Build a small feature/response triple with a learnable signal.

    :param n_cells: number of cell lines
    :param n_drugs: number of drugs
    :param genes: gene-expression dimensionality
    :param bits: fingerprint dimensionality
    :returns: (response dataset, cell line features, drug features)
    """
    rng = np.random.default_rng(0)
    cells = [f"CL{i}" for i in range(n_cells)]
    drugs = [f"D{i}" for i in range(n_drugs)]
    cl_features = FeatureDataset({cl: {"gene_expression": rng.normal(size=genes)} for cl in cells})
    drug_features = FeatureDataset({d: {"fingerprints": rng.integers(0, 2, size=bits).astype(float)} for d in drugs})

    cl_ids, drug_ids = [], []
    for cl in cells:
        for drug in drugs:
            cl_ids.append(cl)
            drug_ids.append(drug)
    cell_signal = {cl: rng.normal() for cl in cells}
    response = np.array([cell_signal[cl] + 0.1 * rng.normal() for cl in cl_ids])
    output = DrugResponseDataset(
        response=response,
        cell_line_ids=np.array(cl_ids),
        drug_ids=np.array(drug_ids),
        dataset_name="TOY_GB",
    )
    return output, cl_features, drug_features


def test_model_is_registered() -> None:
    """The model has to be reachable through the registry used by the CLI."""
    assert MODEL_FACTORY["GradientBoostingReproducible"] is GradientBoostingReproducible
    assert GradientBoostingReproducible.get_model_name() == "GradientBoostingReproducible"


def test_grid_matches_the_upstream_gradient_boosting_grid() -> None:
    """Only the stochastic settings may differ from the upstream baseline grid."""
    reproducible = GradientBoostingReproducible.get_hyperparameter_set()
    upstream = GradientBoosting.get_hyperparameter_set()
    assert len(reproducible) == len(upstream) == 6
    for ours, theirs in zip(reproducible, upstream, strict=True):
        for key in ["max_iter", "learning_rate", "max_depth", "cell_line_views", "drug_views", "gene_list"]:
            assert ours[key] == theirs[key]
        assert ours["random_state"] == 42
        assert ours["early_stopping"] is True


def test_build_model_fixes_the_random_state() -> None:
    """Without a fixed random state, early stopping picks a different internal split every fit."""
    model = GradientBoostingReproducible()
    model.build_model(dict(_HPAMS))
    assert model.model.random_state == 42
    assert model.model.early_stopping is True


def test_repeated_fits_are_identical() -> None:
    """Two fits on identical data must produce identical predictions."""
    output, cl_features, drug_features = _toy_data()
    predictions = []
    for _ in range(2):
        model = GradientBoostingReproducible()
        model.build_model(dict(_HPAMS))
        model.train(output.copy(), cl_features.copy(), drug_features.copy())
        predictions.append(
            model.predict(
                cell_line_ids=output.cell_line_ids.copy(),
                drug_ids=output.drug_ids.copy(),
                cell_line_input=cl_features.copy(),
                drug_input=drug_features.copy(),
            )
        )
    assert np.array_equal(predictions[0], predictions[1])


def test_predictions_follow_the_requested_identifier_order() -> None:
    """Shuffling the requested rows may only permute the predictions."""
    output, cl_features, drug_features = _toy_data()
    model = GradientBoostingReproducible()
    model.build_model(dict(_HPAMS))
    model.train(output.copy(), cl_features.copy(), drug_features.copy())

    order = np.arange(len(output))
    np.random.default_rng(1).shuffle(order)
    base = model.predict(
        cell_line_ids=output.cell_line_ids.copy(),
        drug_ids=output.drug_ids.copy(),
        cell_line_input=cl_features.copy(),
        drug_input=drug_features.copy(),
    )
    shuffled = model.predict(
        cell_line_ids=output.cell_line_ids[order],
        drug_ids=output.drug_ids[order],
        cell_line_input=cl_features.copy(),
        drug_input=drug_features.copy(),
    )
    assert np.allclose(shuffled, base[order])


def test_unseen_cell_line_is_predicted_without_error() -> None:
    """LCO prediction has to work for cell lines that were not in training."""
    output, cl_features, drug_features = _toy_data()
    model = GradientBoostingReproducible()
    model.build_model(dict(_HPAMS))
    model.train(output.copy(), cl_features.copy(), drug_features.copy())

    unseen = FeatureDataset({**cl_features.features, "UNSEEN": {"gene_expression": np.zeros(6)}})
    predictions = model.predict(
        cell_line_ids=np.array(["UNSEEN"]),
        drug_ids=np.array(["D0"]),
        cell_line_input=unseen,
        drug_input=drug_features.copy(),
    )
    assert predictions.shape == (1,)
    assert np.isfinite(predictions[0])


def test_missing_drug_feature_raises() -> None:
    """A drug without a feature vector must fail loudly instead of silently predicting."""
    output, cl_features, drug_features = _toy_data()
    model = GradientBoostingReproducible()
    model.build_model(dict(_HPAMS))
    model.train(output.copy(), cl_features.copy(), drug_features.copy())

    with pytest.raises(AssertionError, match="not in the FeatureDataset"):
        model.predict(
            cell_line_ids=np.array(["CL0"]),
            drug_ids=np.array(["NOT_A_DRUG"]),
            cell_line_input=cl_features.copy(),
            drug_input=drug_features.copy(),
        )
