"""Regression test for PharmaFormerModel.predict() gene expression preprocessing."""

import tempfile

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.PharmaFormer.model_utils import CombinedModel
from drevalpy.models.PharmaFormer.pharmaformer import PharmaFormerModel


def _build_fitted_model() -> PharmaFormerModel:
    """
    Builds a PharmaFormerModel with fitted scalers and a small randomly initialized network.

    :return: a PharmaFormerModel ready for predict() calls
    """
    rng = np.random.default_rng(0)
    gene_input_size = 10

    model = PharmaFormerModel()
    model.hyperparameters = {"batch_size": 64}

    train_gene_features = rng.normal(size=(50, gene_input_size))
    model.gene_expression_scaler = StandardScaler().fit(train_gene_features)
    model.gene_expression_normalizer = MinMaxScaler().fit(model.gene_expression_scaler.transform(train_gene_features))

    torch.manual_seed(0)
    model.model = CombinedModel(
        gene_input_size=gene_input_size,
        gene_hidden_size=8,
        drug_hidden_size=8,
        feature_dim=4,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    )
    return model


def test_predict_is_invariant_to_cell_line_repetition() -> None:
    """A cell line's prediction must not change depending on how many times it repeats in the request."""
    model = _build_fitted_model()
    rng = np.random.default_rng(1)

    cell_line_features = FeatureDataset(
        features={
            "cellA": {"gene_expression": rng.normal(size=10).astype(np.float32)},
            "cellB": {"gene_expression": rng.normal(size=10).astype(np.float32)},
        }
    )
    drug_features = FeatureDataset(
        features={f"drug{i}": {"bpe_smiles": rng.normal(size=128).astype(np.float32)} for i in range(5)}
    )

    # Predict each (cell line, drug) pair in isolation, one request per pair.
    isolated_predictions = {}
    for cell_line_id, drug_id in [("cellA", "drug0"), ("cellA", "drug1"), ("cellB", "drug3")]:
        isolated_predictions[(cell_line_id, drug_id)] = model.predict(
            cell_line_ids=np.array([cell_line_id]),
            drug_ids=np.array([drug_id]),
            cell_line_input=cell_line_features.copy(),
            drug_input=drug_features.copy(),
        )[0]

    # Predict the same pairs again, but this time cellA also occurs repeatedly against
    # other drugs in the same request, as happens whenever a held-out cell line is
    # tested against many drugs in one fold.
    repeated_cell_line_ids = np.array(["cellA", "cellA", "cellA", "cellB", "cellA"])
    repeated_drug_ids = np.array(["drug0", "drug1", "drug2", "drug3", "drug4"])
    repeated_predictions = model.predict(
        cell_line_ids=repeated_cell_line_ids,
        drug_ids=repeated_drug_ids,
        cell_line_input=cell_line_features.copy(),
        drug_input=drug_features.copy(),
    )

    assert repeated_predictions.shape == (5,)
    # Batch-of-1 vs batch-of-5 forward passes are not guaranteed bit-identical in float32,
    # so this compares with a looser tolerance than the exact scaling check below.
    assert np.allclose(isolated_predictions[("cellA", "drug0")], repeated_predictions[0], atol=1e-4)
    assert np.allclose(isolated_predictions[("cellA", "drug1")], repeated_predictions[1], atol=1e-4)
    assert np.allclose(isolated_predictions[("cellB", "drug3")], repeated_predictions[3], atol=1e-4)


def test_predict_scales_gene_expression_only_once_regardless_of_repeat_count() -> None:
    """The gene expression fed into the network must be the once-transformed vector, not repeatedly rescaled."""
    model = _build_fitted_model()
    rng = np.random.default_rng(2)

    raw_gene_expr = rng.normal(size=10).astype(np.float32)
    cell_line_features = FeatureDataset(features={"cellA": {"gene_expression": raw_gene_expr}})
    drug_features = FeatureDataset(features={"drug0": {"bpe_smiles": rng.normal(size=128).astype(np.float32)}})

    assert model.gene_expression_scaler is not None
    assert model.gene_expression_normalizer is not None
    assert model.model is not None
    expected = model.gene_expression_normalizer.transform(
        model.gene_expression_scaler.transform(raw_gene_expr.reshape(1, -1))
    ).flatten()

    seen_gene_inputs = []
    handle = model.model.feature_extractor.gene_fc1.register_forward_pre_hook(
        lambda module, inputs: seen_gene_inputs.append(inputs[0].clone())
    )
    try:
        for repeat_count in (1, 3, 10):
            cell_line_ids = np.array(["cellA"] * repeat_count)
            drug_ids = np.array(["drug0"] * repeat_count)
            model.predict(
                cell_line_ids=cell_line_ids,
                drug_ids=drug_ids,
                cell_line_input=cell_line_features.copy(),
                drug_input=drug_features.copy(),
            )
    finally:
        handle.remove()

    assert len(seen_gene_inputs) == 3
    for repeat_count, gene_input in zip((1, 3, 10), seen_gene_inputs):
        actual = gene_input.numpy()
        assert actual.shape == (repeat_count, 10)
        for row in actual:
            assert np.allclose(row, expected, atol=1e-6)


def test_train_fits_gene_expression_scaler_without_duplicate_weighting() -> None:
    """The scaler must be fit once per cell line, not weighted by how many drugs each was tested against."""
    model = PharmaFormerModel()
    model.hyperparameters = {
        "gene_hidden_size": 8,
        "drug_hidden_size": 8,
        "feature_dim": 4,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 1,
        "patience": 1,
    }

    rng = np.random.default_rng(3)
    cell_a_expr = rng.normal(size=10).astype(np.float32)
    cell_b_expr = rng.normal(size=10).astype(np.float32)
    cell_line_input = FeatureDataset(
        features={
            "cellA": {"gene_expression": cell_a_expr},
            "cellB": {"gene_expression": cell_b_expr},
        }
    )
    drug_input = FeatureDataset(
        features={f"drug{i}": {"bpe_smiles": rng.normal(size=128).astype(np.float32)} for i in range(3)}
    )

    # cellA is tested against three drugs, cellB against only one: a fit on the raw
    # response-level array would weight cellA's expression vector 3x as heavily as cellB's.
    output = DrugResponseDataset(
        response=rng.normal(size=4).astype(np.float32),
        cell_line_ids=np.array(["cellA", "cellA", "cellA", "cellB"]),
        drug_ids=np.array(["drug0", "drug1", "drug2", "drug0"]),
    )
    output_earlystopping = DrugResponseDataset(
        response=rng.normal(size=2).astype(np.float32),
        cell_line_ids=np.array(["cellA", "cellB"]),
        drug_ids=np.array(["drug1", "drug0"]),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model.train(
            output=output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            output_earlystopping=output_earlystopping,
            model_checkpoint_dir=tmpdir,
        )

    expected_mean = np.mean([cell_a_expr, cell_b_expr], axis=0)
    assert model.gene_expression_scaler is not None
    assert np.allclose(model.gene_expression_scaler.mean_, expected_mean, atol=1e-6)
