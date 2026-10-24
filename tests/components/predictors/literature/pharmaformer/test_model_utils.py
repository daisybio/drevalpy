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
