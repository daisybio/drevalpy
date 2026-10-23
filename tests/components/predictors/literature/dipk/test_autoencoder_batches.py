"""Tests for the DIPK autoencoder's batch-size-dependent helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from drevalpy.components.predictors.literature.dipk._autoencoder_batches import (
    avoid_singleton_batch,
    validation_loss,
)
from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import (
    GeneExpressionDecoder,
    GeneExpressionEncoder,
)

INPUT_DIM = 6
_CPU = torch.device("cpu")


def _eval_models() -> tuple[GeneExpressionEncoder, GeneExpressionDecoder]:
    torch.manual_seed(0)
    encoder = GeneExpressionEncoder(INPUT_DIM)
    decoder = GeneExpressionDecoder(INPUT_DIM)
    encoder.eval()
    decoder.eval()
    return encoder, decoder


@pytest.mark.parametrize("batch_size", [1, 3, 7, 10, 1024])
def test_chunked_validation_loss_matches_full_mse(batch_size: int) -> None:
    """The chunked pass returns what MSELoss over the whole matrix returns, also for non-dividing sizes."""
    encoder, decoder = _eval_models()
    validation = torch.randn(10, INPUT_DIM)

    with torch.no_grad():
        expected = nn.MSELoss()(decoder(encoder(validation)), validation).item()

    assert validation_loss(encoder, decoder, validation, batch_size, _CPU) == pytest.approx(expected, rel=1e-5)


def test_validation_loss_leaves_the_input_untouched() -> None:
    encoder, decoder = _eval_models()
    validation = torch.randn(5, INPUT_DIM)
    before = validation.clone()

    validation_loss(encoder, decoder, validation, 2, _CPU)

    assert torch.equal(validation, before)


def test_validation_loss_of_an_empty_matrix_is_nan() -> None:
    encoder, decoder = _eval_models()

    assert np.isnan(validation_loss(encoder, decoder, torch.empty(0, INPUT_DIM), 4, _CPU))


@pytest.mark.parametrize(
    ("n_train", "batch_size", "moved"),
    [(5, 4, True), (9, 4, True), (1025, 1024, True), (4, 4, False), (6, 4, False), (8, 1, False), (1, 1024, False)],
)
def test_avoid_singleton_batch_moves_the_last_row_only_for_a_trailing_batch_of_one(
    n_train: int, batch_size: int, moved: bool
) -> None:
    train = np.arange(n_train * 2, dtype=np.float32).reshape(n_train, 2)
    validation = np.full((3, 2), -1.0, dtype=np.float32)

    new_train, new_validation = avoid_singleton_batch(train, validation, batch_size)

    if moved:
        np.testing.assert_array_equal(new_train, train[:-1])
        np.testing.assert_array_equal(new_validation, np.concatenate([validation, train[-1:]]))
    else:
        assert new_train is train
        assert new_validation is validation
