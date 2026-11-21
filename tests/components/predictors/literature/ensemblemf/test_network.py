"""Tests for the EnsembleMF two-tower network and member training loop."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy.components.predictors.literature.ensemblemf.network import MFNet, MFNetConfig, ResidualEncoder
from drevalpy.components.predictors.literature.ensemblemf.training import (
    MemberData,
    MemberSettings,
    PairTensors,
    train_member,
    validation_mse,
)

CONFIG = MFNetConfig(sample_in_dim=7, drug_in_dim=5, n_drugs=3, hidden_dim=8, emb_dim=4, mlp_hidden=6)


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


@pytest.mark.parametrize("n_layers", [0, 2, 3])
def test_encoder_depth_follows_n_layers(n_layers: int) -> None:
    encoder = ResidualEncoder(in_dim=7, hidden_dim=8, emb_dim=4, n_layers=n_layers, dropout=0.0)

    assert encoder(torch.randn(2, 7)).shape == (2, 4)
    assert len(encoder.blocks) == n_layers


def _pairs(sample_idx: list[int], drug_idx: list[int], response: list[float]) -> PairTensors:
    return PairTensors(torch.tensor(sample_idx), torch.tensor(drug_idx), torch.tensor(response, dtype=torch.float32))


def _data() -> MemberData:
    generator = torch.Generator().manual_seed(0)
    sample_idx = [0, 1, 2, 3, 0, 1, 2, 3]
    drug_idx = [0, 0, 0, 0, 1, 1, 1, 1]
    return MemberData(
        x_sample=torch.randn(4, 6, generator=generator),
        x_drug=torch.randn(2, 5, generator=generator),
        train=_pairs(sample_idx, drug_idx, [0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0]),
        validation=_pairs([1, 2], [1, 0], [2.0, 2.0]),
    )


def _net() -> MFNet:
    torch.manual_seed(1)
    return MFNet(MFNetConfig(sample_in_dim=6, drug_in_dim=5, n_drugs=2, hidden_dim=8, emb_dim=4, mlp_hidden=4))


SETTINGS = MemberSettings(learning_rate=1e-2, weight_decay=0.0, batch_size=4, max_epochs=30, patience=30)


def test_validation_mse_is_infinite_without_validation_pairs() -> None:
    assert len(_pairs([0, 1], [0, 0], [1.0, 2.0])) == 2
    data = _data()
    empty = MemberData(data.x_sample, data.x_drug, data.train, _pairs([], [], []))

    assert validation_mse(_net(), empty) == float("inf")


def test_training_restores_the_best_validation_weights() -> None:
    data = _data()
    net = _net()
    before = validation_mse(net, data)

    train_member(net, data, SETTINGS)

    assert validation_mse(net, data) < before


def test_training_is_reproducible_under_a_fixed_seed() -> None:
    data = _data()
    results = []
    for _ in range(2):
        torch.manual_seed(5)
        net = _net()
        torch.manual_seed(5)
        train_member(net, data, SETTINGS)
        results.append(torch.cat([p.detach().flatten() for p in net.parameters()]).numpy())

    np.testing.assert_array_equal(results[0], results[1])


def test_training_stops_when_patience_runs_out_and_restores_the_best_state() -> None:
    data = _data()
    net = _net()
    settings = MemberSettings(learning_rate=0.0, weight_decay=0.0, batch_size=4, max_epochs=50, patience=1)
    initial = {key: value.clone() for key, value in net.state_dict().items()}

    train_member(net, data, settings)

    for key, value in net.state_dict().items():
        torch.testing.assert_close(value, initial[key])


@pytest.mark.parametrize("use_embedding", [True, False])
def test_training_runs_with_and_without_the_id_embedding(use_embedding: bool) -> None:
    torch.manual_seed(0)
    config = MFNetConfig(
        sample_in_dim=6, drug_in_dim=5, n_drugs=2, hidden_dim=8, emb_dim=4, use_drug_id_embedding=use_embedding
    )
    data = _data()

    train_member(MFNet(config), data, MemberSettings(1e-2, 0.0, 4, 3, 3))
