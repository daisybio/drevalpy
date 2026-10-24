"""Tests for the PharmaFormer predictor and its feature extractor and transformer regressor."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from drevalpy.components.predictors.literature.pharmaformer.model_utils import (
    CombinedModel,
    FeatureExtractor,
    TransModel,
)
from drevalpy.components.predictors.literature.pharmaformer.predictor import (
    PharmaFormerPredictor,
    _build_combined_model,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.utils.torch_io import save_trusted_mapping
from tests.components.predictors.literature._helpers import (
    assert_frozen_reference,
    assert_state_round_trip,
    assert_trainable_parameters_changed,
    assert_training_predictions,
    finite_predictions,
    seed_literature,
    train_validation_batch,
    trainable_parameter_vector,
)


def test_registry_name_and_set_state_requires_the_model_state() -> None:
    assert get_predictor("pharmaFormer") is PharmaFormerPredictor
    payload = save_trusted_mapping(
        {"predictor_hyperparameters": PharmaFormerPredictor.get_default_hyperparameters(), "gene_input_size": 6}
    )

    with pytest.raises(PredictorStateError, match="missing gene_input_size or model_state"):
        PharmaFormerPredictor().set_state({"payload": payload})


def _trainable_batch(tmp_path) -> ModelInputBatch:
    rng = np.random.default_rng(987)
    return train_validation_batch(
        {"gene_expression": numeric_feature_block(rng.normal(size=(4, 6)).astype(np.float32))},
        {"bpe_smiles": numeric_feature_block(rng.normal(size=(2, 128)).astype(np.float32))},
        tmp_path,
    )


class TestTinyPharmaFormerFit:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_updates_parameters_and_round_trips_state(self, seed: int, tmp_path) -> None:
        hyperparameters = {
            "epochs": 1,
            "patience": 1,
            "batch_size": 4,
            "lr": 0.001,
            "dropout": 0.0,
            "gene_hidden_size": 16,
            "drug_hidden_size": 16,
            "feature_dim": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
        }
        batch = _trainable_batch(tmp_path)
        seed_literature(seed)
        device = PharmaFormerPredictor(hyperparameters)._device
        initial = trainable_parameter_vector(_build_combined_model(6, hyperparameters, device))

        seed_literature(seed)
        predictor = PharmaFormerPredictor(hyperparameters=hyperparameters)
        predictor.fit(batch)
        assert predictor._model is not None
        assert_trainable_parameters_changed(initial, predictor._model)
        expected = finite_predictions(predictor, batch)
        assert_training_predictions("pharmaformer", seed, expected)
        assert_frozen_reference("pharmaformer", seed)

        assert_state_round_trip(predictor, batch, expected)


SMILES_DIM = 128


def _combined(*, gene_input_size: int = 6) -> CombinedModel:
    model = CombinedModel(
        gene_input_size=gene_input_size,
        gene_hidden_size=8,
        drug_hidden_size=8,
        feature_dim=4,
        nhead=2,
        num_layers=1,
        dim_feedforward=8,
        dropout=0.0,
    )
    model.eval()
    return model


def test_feature_extractor_concatenates_the_branches_after_a_relu() -> None:
    extractor = FeatureExtractor(gene_input_size=6, gene_hidden_size=8, drug_hidden_size=5)
    extractor.eval()

    output = extractor(torch.randn(3, 6), torch.randn(3, SMILES_DIM))

    assert output.shape == (3, 13)
    assert (output >= 0).all()
    assert extractor.smiles_fc.in_features == SMILES_DIM


def test_trans_model_reduces_a_sequence_to_one_scalar_per_row() -> None:
    model = TransModel(feature_dim=4, nhead=2, seq_len=3, dim_feedforward=8, dropout=0.0, num_layers=2)
    model.eval()

    assert model(torch.randn(5, 3, 4)).shape == (5, 1)
    assert model.transformer_encoder.num_layers == 2
    assert next(layer for layer in model.output if isinstance(layer, nn.Linear)).in_features == 3 * 4


def test_combined_model_predicts_one_deterministic_scalar_per_pair_and_trains_the_gene_branch() -> None:
    model = _combined()
    assert (model.seq_len, model.feature_dim) == (4, 4)
    gene, smiles = torch.randn(3, 6), torch.randn(3, SMILES_DIM)

    with torch.no_grad():
        first = model(gene, smiles)
        second = model(gene, smiles)
        single = model(gene[:1], smiles[:1])

    assert first.shape == (3, 1)
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second)
    assert single.shape == (1, 1)

    model.train()
    model(gene, smiles).sum().backward()
    assert model.feature_extractor.gene_fc1.weight.grad is not None
