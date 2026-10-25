"""Tests for the DIPK block predictor and its attention, dense and combined network modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from drevalpy._utils.torch_io import save_trusted_mapping
from drevalpy.components.predictors.literature.dipk.attention_utils import MultiHeadAttentionLayer
from drevalpy.components.predictors.literature.dipk.model_utils import (
    DEVICE,
    AttentionLayer,
    DenseLayers,
    Predictor,
    features_dim_bionic,
    features_dim_gene,
)
from drevalpy.components.predictors.literature.dipk.predictor import (
    DIPKPredictor,
    _CollateFn,
    _DIPKDataset,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import numeric_feature_block, ragged_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from tests.components.predictors.literature._helpers import (
    assert_frozen_reference,
    assert_state_round_trip,
    assert_trainable_parameters_changed,
    assert_training_predictions,
    finite_predictions,
    seed_literature,
    train_validation_batch,
    trainable_parameter_vector,
    two_by_two_batch,
)


def _dipk_batch() -> ModelInputBatch:
    molgnet = np.empty(2, dtype=object)
    molgnet[:] = [
        np.arange(6, dtype=np.float32).reshape(3, 2),
        np.arange(4, dtype=np.float32).reshape(2, 2),
    ]
    return two_by_two_batch(
        sample_blocks={
            "gene_expression": numeric_feature_block(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)),
            "bionic_features": numeric_feature_block(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
        },
        drug_blocks={"molgnet_features": ragged_feature_block(molgnet)},
        checkpoint_dir=Path(tempfile.mkdtemp()),
    )


def _trainable_dipk_batch(tmp_path: Path) -> ModelInputBatch:
    rng = np.random.default_rng(321)
    molgnet = np.empty(2, dtype=object)
    molgnet[:] = [rng.normal(size=(2, 768)).astype(np.float32), rng.normal(size=(3, 768)).astype(np.float32)]
    return train_validation_batch(
        {
            "gene_expression": numeric_feature_block(rng.normal(size=(4, 512)).astype(np.float32)),
            "bionic_features": numeric_feature_block(rng.normal(size=(4, 512)).astype(np.float32)),
        },
        {"molgnet_features": ragged_feature_block(molgnet)},
        tmp_path,
    )


def test_registry_name_and_default_hyperparameters() -> None:
    assert get_predictor("dipk") is DIPKPredictor
    assert {"batch_size", "lr", "epochs", "patience"} <= set(DIPKPredictor.get_default_hyperparameters())


class TestFrozenDIPKNumericalReference:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_v2(self, seed: int) -> None:
        assert_frozen_reference("dipk", seed)


class TestTinyDIPKFit:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_updates_parameters_and_round_trips_state(self, seed: int, tmp_path: Path) -> None:
        hyperparameters = {
            "batch_size": 4,
            "lr": 0.001,
            "heads": 1,
            "fc_layer_num": 3,
            "fc_layer_dim": [8, 8, 8, 8, 8, 8],
            "dropout_rate": 0.0,
            "epochs": 1,
            "patience": 1,
        }
        seed_literature(seed)
        initial_predictor = DIPKPredictor(hyperparameters)
        initial = trainable_parameter_vector(initial_predictor._build_model())

        seed_literature(seed)
        predictor = DIPKPredictor(hyperparameters)
        batch = _trainable_dipk_batch(tmp_path)
        predictor.fit(batch)
        assert predictor._model is not None
        assert_trainable_parameters_changed(initial, predictor._model)

        predictions = finite_predictions(predictor, batch)
        assert_training_predictions("dipk", seed, predictions)

        assert_state_round_trip(predictor, batch, predictions, rtol=1e-6, atol=1e-7)


def test_build_pairs_selects_entity_rows_and_adds_the_target_for_a_response() -> None:
    predictor = DIPKPredictor()
    batch = _dipk_batch()

    pairs = predictor._build_pairs(batch.sample_pair_idx, batch.drug_pair_idx, batch)

    assert len(pairs) == 4
    assert set(pairs[0]) == {"molgnet_features", "gene_expression", "bionic_features"}
    torch.testing.assert_close(pairs[2]["gene_expression"], torch.tensor([0.3, 0.4]))
    assert pairs[2]["molgnet_features"].shape == (3, 2)
    with_target = predictor._build_pairs(
        batch.sample_pair_idx, batch.drug_pair_idx, batch, response=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    )
    assert with_target[3]["ic50"].tolist() == [4.0]


def test_collate_pads_ragged_atoms_masks_them_and_adds_targets_only_in_training() -> None:
    pairs = [
        {
            "molgnet_features": torch.ones(3, 2),
            "gene_expression": torch.tensor([0.1, 0.2]),
            "bionic_features": torch.tensor([1.0, 2.0]),
            "ic50": torch.tensor([5.0]),
        },
        {
            "molgnet_features": torch.ones(1, 2),
            "gene_expression": torch.tensor([0.3, 0.4]),
            "bionic_features": torch.tensor([3.0, 4.0]),
            "ic50": torch.tensor([6.0]),
        },
    ]

    predict = _CollateFn(train=False)(pairs)
    train = _CollateFn(train=True)(pairs)

    assert predict["molgnet_features"].shape == (2, 3, 2)
    assert predict["molgnet_features"][1, 1:].abs().sum().item() == 0.0
    assert predict["molgnet_mask"].dtype is torch.bool
    assert predict["molgnet_mask"][1].tolist() == [True, False, False]
    assert train["ic50_values"].flatten().tolist() == [5.0, 6.0]
    assert "ic50_values" not in predict


def test_dataset_drives_a_dataloader_without_subclassing_torch_dataset() -> None:
    from torch.utils.data import DataLoader

    second = {"a": torch.ones(1)}
    dataset = _DIPKDataset([{"a": torch.zeros(1)}, second])

    assert len(dataset) == 2
    assert dataset[1] is second
    assert len(next(iter(DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=list)))) == 2


def test_guards_and_unfitted_state() -> None:
    predictor = DIPKPredictor()
    predictions = predictor._predict(_dipk_batch())
    assert np.isnan(predictions).all()
    assert len(predictions) == 4
    assert predictor.is_fitted() is False
    assert predictor.get_state() == {}
    with pytest.raises(ValueError, match="early stopping data"):
        predictor._fit(_dipk_batch())


def test_set_state_rejects_a_missing_payload_or_model_state() -> None:
    with pytest.raises(PredictorStateError, match="payload byte blob"):
        DIPKPredictor().set_state({"payload": "not-bytes"})
    payload = save_trusted_mapping({"predictor_hyperparameters": DIPKPredictor.get_default_hyperparameters()})
    with pytest.raises(PredictorStateError, match="missing model_state"):
        DIPKPredictor().set_state({"payload": payload})


FC_LAYER_DIM = [16, 8, 4, 4, 4, 4]
HIDDEN_DIM = 768


def _molgnet(batch_size: int, seq_len: int) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, HIDDEN_DIM)


def test_device_and_feature_dims() -> None:
    assert DEVICE.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert features_dim_gene == features_dim_bionic == 512


def test_attention_layer_projects_to_the_hidden_dim_and_squeezes_a_single_row() -> None:
    layer = AttentionLayer(heads=1)
    layer.eval()

    def run(batch_size: int, seq_len: int) -> torch.Tensor:
        return layer(
            _molgnet(batch_size, seq_len),
            torch.ones(batch_size, seq_len),
            torch.randn(batch_size, features_dim_gene),
            torch.randn(batch_size, features_dim_bionic),
        )

    assert run(2, 5).shape == (2, HIDDEN_DIM)
    assert run(1, 3).shape == (HIDDEN_DIM,)
    with pytest.raises(ValueError, match="divisible by the number of heads"):
        AttentionLayer(heads=5)


def test_dense_layers_reduce_to_one_scalar_per_row_and_size_their_head() -> None:
    dense = DenseLayers(fc_layer_num=3, fc_layer_dim=FC_LAYER_DIM, dropout_rate=0.0)
    dense.eval()
    gene, bionic = torch.randn(4, features_dim_gene), torch.randn(4, features_dim_bionic)

    assert dense(torch.randn(4, HIDDEN_DIM), gene, bionic).shape == (4, 1)
    assert dense(torch.randn(HIDDEN_DIM), gene[:1], bionic[:1]).shape == (1, 1)
    assert isinstance(dense.fc_output, nn.Linear)
    assert (dense.fc_output.in_features, dense.fc_output.out_features) == (FC_LAYER_DIM[1], 1)
    dropout = DenseLayers(fc_layer_num=4, fc_layer_dim=FC_LAYER_DIM, dropout_rate=0.25)
    assert {layer.p for layer in dropout.dropout_layers} == {0.25}
    assert len(dropout.dropout_layers) == 4


def test_predictor_scores_one_response_per_pair_and_trains_the_attention_stage() -> None:
    predictor = Predictor(heads=1, fc_layer_num=3, fc_layer_dim=FC_LAYER_DIM, dropout_rate=0.0)
    assert isinstance(predictor.attention_layer, AttentionLayer)
    assert isinstance(predictor.dense_layers, DenseLayers)

    def run(seq_len: int) -> torch.Tensor:
        return predictor(
            _molgnet(2, seq_len),
            torch.randn(2, features_dim_gene),
            torch.randn(2, features_dim_bionic),
            torch.ones(2, seq_len),
        )

    predictor.eval()
    with torch.no_grad():
        output = run(4)
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()

    predictor.train()
    run(3).sum().backward()
    assert predictor.attention_layer.fc_layer_0.weight.grad is not None


def _layer(*, hid_dim: int = 8, n_heads: int = 2, dropout: float = 0.0) -> MultiHeadAttentionLayer:
    layer = MultiHeadAttentionLayer(hid_dim=hid_dim, n_heads=n_heads, dropout=dropout, device="cpu")
    layer.eval()
    return layer


def test_layer_rejects_indivisible_heads_and_splits_the_hidden_dim() -> None:
    with pytest.raises(ValueError, match="divisible by the number of heads"):
        MultiHeadAttentionLayer(hid_dim=10, n_heads=4, dropout=0.0, device="cpu")
    layer = _layer(hid_dim=16, n_heads=4)
    assert layer.head_dim == 4
    assert layer.scale.item() == pytest.approx(2.0)
    assert _layer(hid_dim=6, n_heads=1).head_dim == 6


def test_output_keeps_the_query_length_and_weights_sum_to_one_and_are_differentiable() -> None:
    layer = _layer(hid_dim=8, n_heads=2)
    query = torch.randn(3, 1, 8)
    key = torch.randn(3, 5, 8)

    output, attention = layer(query, key, key)

    assert output.shape == (3, 1, 8)
    assert attention.shape == (3, 2, 1, 5)
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(3, 2, 1))
    output.sum().backward()
    assert layer.fc_q.weight.grad is not None


def test_attention_mask_removes_masked_keys_from_the_weights() -> None:
    layer = _layer(hid_dim=8, n_heads=2)
    key = torch.randn(1, 4, 8)
    mask = torch.tensor([[[[1, 1, 0, 0]]]])

    _, attention = layer(torch.randn(1, 1, 8), key, key, mask)

    assert attention[..., 2:].abs().max().item() == pytest.approx(0.0)
    torch.testing.assert_close(attention[..., :2].sum(dim=-1), torch.ones(1, 2, 1))
