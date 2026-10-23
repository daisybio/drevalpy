"""Tests for the PaccMann network."""

from __future__ import annotations

import pytest
import torch

from drevalpy.components.predictors.literature.paccmann.network import (
    ContextAttentionLayer,
    PaccMannConfig,
    PaccMannNetwork,
)

SMALL = {
    "smiles_vocabulary_size": 12,
    "smiles_embedding_size": 4,
    "smiles_padding_length": 16,
    "number_of_genes": 7,
    "molecule_heads": [1, 2, 1, 1],
    "gene_heads": [1, 1, 1, 1],
    "filters": [3, 3, 3],
    "smiles_attention_size": 5,
    "stacked_dense_hidden_sizes": [8, 4],
    "dropout": 0.0,
    "batch_norm": True,
}


def _network() -> PaccMannNetwork:
    return PaccMannNetwork(PaccMannConfig.from_hyperparameters(SMALL))


def test_config_resolves_the_default_kernel_windows() -> None:
    config = PaccMannConfig.from_hyperparameters(SMALL)

    assert config.resolved_kernel_sizes == [(3, 4), (5, 4), (11, 4)]


def test_config_ignores_keys_it_does_not_know() -> None:
    config = PaccMannConfig.from_hyperparameters({**SMALL, "epochs": 10, "learning_rate": 0.1})

    assert config.number_of_genes == 7


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"gene_heads": [1, 1]}, "same length"),
        ({"kernel_sizes": [(3, 4), (5, 4)]}, "filters and kernel_sizes"),
        ({"filters": [3, 3], "kernel_sizes": [(3, 4), (5, 4)]}, "one more entry than filters"),
    ],
)
def test_config_rejects_inconsistent_lengths(override: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PaccMannConfig.from_hyperparameters({**SMALL, **override})


def test_attention_layer_weights_sum_to_one_over_the_reference() -> None:
    layer = ContextAttentionLayer(4, 6, 1, 7, attention_size=5, temperature=1.0)

    attended, alphas = layer(torch.randn(3, 6, 4), torch.randn(3, 7, 1))

    assert attended.shape == (3, 4)
    assert alphas.shape == (3, 6)
    torch.testing.assert_close(alphas.sum(dim=1), torch.ones(3))


def test_attention_layer_can_keep_the_sequence_axis() -> None:
    layer = ContextAttentionLayer(1, 7, 4, 6, attention_size=1, temperature=1.0)

    attended, _ = layer(torch.randn(3, 7, 1), torch.randn(3, 6, 4), average_seq=False)

    assert attended.shape == (3, 7)


def test_a_single_position_context_skips_the_length_projection() -> None:
    layer = ContextAttentionLayer(4, 6, 1, 1, attention_size=5, temperature=1.0)

    assert isinstance(layer.context_hidden_projection, torch.nn.Identity)


def test_network_predicts_one_value_per_row() -> None:
    network = _network().eval()

    output = network(torch.randint(0, 12, (5, 16)), torch.randn(5, 7))

    assert output.shape == (5, 1)
    assert torch.isfinite(output).all()


def test_network_keeps_the_batch_axis_for_a_single_row_in_eval_mode() -> None:
    network = _network().eval()

    assert network(torch.randint(0, 12, (1, 16)), torch.randn(1, 7)).shape == (1, 1)


def test_attention_heads_follow_the_configuration() -> None:
    network = _network()

    assert len(network.molecule_attentions) == sum(SMALL["molecule_heads"])
    assert len(network.gene_attentions) == sum(SMALL["gene_heads"])
    assert len(network.convolutions) == 3


def test_batch_norm_can_be_switched_off() -> None:
    network = PaccMannNetwork(PaccMannConfig.from_hyperparameters({**SMALL, "batch_norm": False}))

    assert isinstance(network.batch_norm, torch.nn.Identity)
