"""Tests for the EnsembleMF member training loop."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy.components.predictors.literature.ensemblemf.network import MFNet, MFNetConfig
from drevalpy.components.predictors.literature.ensemblemf.training import (
    MemberData,
    MemberSettings,
    PairTensors,
    train_member,
    validation_mse,
)


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


def test_pair_tensors_report_their_length() -> None:
    assert len(_pairs([0, 1], [0, 0], [1.0, 2.0])) == 2


def test_validation_mse_is_infinite_without_validation_pairs() -> None:
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
