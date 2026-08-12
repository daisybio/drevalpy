"""Tests for the PharmaFormer feature extractor and transformer regressor."""

from __future__ import annotations

import torch
from torch import nn

from drevalpy.components.predictors.literature.pharmaformer.model_utils import (
    CombinedModel,
    FeatureExtractor,
    TransModel,
)

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


def test_feature_extractor_concatenates_the_gene_and_drug_branches() -> None:
    extractor = FeatureExtractor(gene_input_size=6, gene_hidden_size=8, drug_hidden_size=5)
    extractor.eval()

    output = extractor(torch.randn(3, 6), torch.randn(3, SMILES_DIM))

    assert output.shape == (3, 13)


def test_feature_extractor_output_is_non_negative_after_relu() -> None:
    extractor = FeatureExtractor(gene_input_size=4, gene_hidden_size=4, drug_hidden_size=4)
    extractor.eval()

    output = extractor(torch.randn(2, 4), torch.randn(2, SMILES_DIM))

    assert (output >= 0).all()


def test_feature_extractor_expects_the_fixed_bpe_smiles_width() -> None:
    extractor = FeatureExtractor(gene_input_size=4, gene_hidden_size=4, drug_hidden_size=3)

    assert extractor.smiles_fc.in_features == SMILES_DIM


def test_trans_model_reduces_a_sequence_to_one_scalar_per_row() -> None:
    model = TransModel(feature_dim=4, nhead=2, seq_len=3, dim_feedforward=8, dropout=0.0, num_layers=1)
    model.eval()

    output = model(torch.randn(5, 3, 4))

    assert output.shape == (5, 1)


def test_trans_model_stacks_the_requested_number_of_encoder_layers() -> None:
    model = TransModel(feature_dim=4, nhead=2, seq_len=3, dim_feedforward=8, dropout=0.0, num_layers=2)

    assert model.transformer_encoder.num_layers == 2


def test_trans_model_head_consumes_the_flattened_sequence() -> None:
    model = TransModel(feature_dim=4, nhead=2, seq_len=3, dim_feedforward=8, dropout=0.0, num_layers=1)

    first_linear = next(layer for layer in model.output if isinstance(layer, nn.Linear))

    assert first_linear.in_features == 3 * 4


def test_combined_model_derives_the_sequence_length_from_the_hidden_sizes() -> None:
    model = _combined()

    assert model.seq_len == 4
    assert model.feature_dim == 4


def test_combined_model_predicts_one_scalar_per_pair() -> None:
    model = _combined()

    with torch.no_grad():
        output = model(torch.randn(3, 6), torch.randn(3, SMILES_DIM))

    assert output.shape == (3, 1)
    assert torch.isfinite(output).all()


def test_combined_model_handles_a_single_row_batch() -> None:
    model = _combined()

    with torch.no_grad():
        output = model(torch.randn(1, 6), torch.randn(1, SMILES_DIM))

    assert output.shape == (1, 1)


def test_combined_model_is_deterministic_in_eval_mode() -> None:
    model = _combined()
    gene = torch.randn(2, 6)
    smiles = torch.randn(2, SMILES_DIM)

    with torch.no_grad():
        first = model(gene, smiles)
        second = model(gene, smiles)

    assert torch.allclose(first, second)


def test_combined_model_gradients_reach_the_gene_branch() -> None:
    model = CombinedModel(
        gene_input_size=6,
        gene_hidden_size=8,
        drug_hidden_size=8,
        feature_dim=4,
        nhead=2,
        num_layers=1,
        dim_feedforward=8,
        dropout=0.0,
    )

    model(torch.randn(2, 6), torch.randn(2, SMILES_DIM)).sum().backward()

    assert model.feature_extractor.gene_fc1.weight.grad is not None
