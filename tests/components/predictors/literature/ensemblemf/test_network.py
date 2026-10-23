"""Tests for the EnsembleMF two-tower network."""

from __future__ import annotations

import pytest
import torch

from drevalpy.components.predictors.literature.ensemblemf.network import MFNet, MFNetConfig, ResidualEncoder

CONFIG = MFNetConfig(sample_in_dim=7, drug_in_dim=5, n_drugs=3, hidden_dim=8, emb_dim=4, mlp_hidden=6)


def test_residual_encoder_maps_features_to_the_latent_width() -> None:
    encoder = ResidualEncoder(in_dim=7, hidden_dim=8, emb_dim=4, n_layers=2, dropout=0.0)

    assert encoder(torch.randn(5, 7)).shape == (5, 4)
    assert len(encoder.blocks) == len(encoder.norms) == 2


def test_net_scores_one_response_per_pair() -> None:
    net = MFNet(CONFIG).eval()
    z_sample, z_drug = net.encode(torch.randn(4, 7), torch.randn(3, 5))

    scores = net.score_pairs(z_sample[[0, 1, 3]], z_drug[[2, 0, 1]])

    assert z_sample.shape == (4, 4)
    assert z_drug.shape == (3, 4)
    assert scores.shape == (3,)


def test_encode_adds_the_given_id_embedding_rows() -> None:
    net = MFNet(CONFIG).eval()
    x_sample, x_drug = torch.randn(2, 7), torch.randn(3, 5)
    base = net.encode(x_sample, x_drug, torch.zeros(3, 4))[1]
    rows = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    shifted = net.encode(x_sample, x_drug, rows)[1]
    default = net.encode(x_sample, x_drug)[1]

    torch.testing.assert_close(shifted, base + rows)
    torch.testing.assert_close(default, base + net.drug_id_emb.weight)


def test_net_without_id_embedding_or_head_has_no_such_parameters() -> None:
    config = MFNetConfig(
        sample_in_dim=7,
        drug_in_dim=5,
        n_drugs=3,
        hidden_dim=8,
        emb_dim=4,
        use_drug_id_embedding=False,
        use_mlp_head=False,
    )
    net = MFNet(config).eval()
    z_sample, z_drug = net.encode(torch.randn(2, 7), torch.randn(3, 5), torch.ones(3, 4))

    assert not hasattr(net, "drug_id_emb")
    assert not hasattr(net, "mlp")
    expected = (z_sample[[0]] * z_drug[[1]]).sum(-1, keepdim=True) * net.dot_scale
    expected = expected + net.sample_bias(z_sample[[0]]) + net.drug_bias(z_drug[[1]]) + net.global_bias
    torch.testing.assert_close(net.score_pairs(z_sample[[0]], z_drug[[1]]), expected.squeeze(-1))


def test_drug_id_embedding_starts_near_zero() -> None:
    torch.manual_seed(0)

    assert MFNet(CONFIG).drug_id_emb.weight.abs().max() < 0.1


@pytest.mark.parametrize("n_layers", [0, 3])
def test_encoder_depth_follows_n_layers(n_layers: int) -> None:
    encoder = ResidualEncoder(in_dim=7, hidden_dim=8, emb_dim=4, n_layers=n_layers, dropout=0.0)

    assert encoder(torch.randn(2, 7)).shape == (2, 4)
    assert len(encoder.blocks) == n_layers
