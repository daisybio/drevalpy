"""Tests for the chunked, host-memory validation pass of the DIPK gene expression autoencoder."""

import numpy as np
import pytest
import torch
from torch import nn

from drevalpy.models.DIPK import gene_expression_encoder as gee
from drevalpy.models.DIPK.gene_expression_encoder import (
    GeneExpressionDecoder,
    GeneExpressionEncoder,
    _validation_loss,
    train_gene_expession_autoencoder,
)

_N_GENES = 16
_CPU = torch.device("cpu")


def _models() -> tuple[GeneExpressionEncoder, GeneExpressionDecoder]:
    """
    Build an encoder/decoder pair in eval mode, so dropout and batch norm are deterministic.

    :returns: encoder and decoder
    """
    torch.manual_seed(0)
    encoder = GeneExpressionEncoder(_N_GENES)
    decoder = GeneExpressionDecoder(_N_GENES)
    encoder.eval()
    decoder.eval()
    return encoder, decoder


@pytest.mark.parametrize("batch_size", [1, 3, 7, 10, 1024])
def test_chunked_validation_loss_matches_full_mse(batch_size: int) -> None:
    """The chunked pass has to return exactly what MSELoss over the whole matrix returns.

    :param batch_size: number of rows evaluated at a time, deliberately including sizes that do
        not divide the number of validation rows
    """
    encoder, decoder = _models()
    validation = torch.randn(10, _N_GENES)

    with torch.no_grad():
        expected = nn.MSELoss()(decoder(encoder(validation)), validation).item()

    assert _validation_loss(encoder, decoder, validation, batch_size, _CPU) == pytest.approx(expected, rel=1e-5)


def test_validation_loss_leaves_the_input_untouched() -> None:
    """The validation matrix is only read, the chunks are copies moved to the device."""
    encoder, decoder = _models()
    validation = torch.randn(5, _N_GENES)
    before = validation.clone()

    _validation_loss(encoder, decoder, validation, 2, _CPU)

    assert validation.device == _CPU
    assert torch.equal(validation, before)


def test_validation_loss_of_an_empty_matrix_is_nan() -> None:
    """An empty early stopping set must not raise a ZeroDivisionError."""
    encoder, decoder = _models()

    assert np.isnan(_validation_loss(encoder, decoder, torch.empty(0, _N_GENES), 4, _CPU))


def test_training_keeps_the_full_matrices_in_host_memory(monkeypatch) -> None:
    """Only mini batches may be moved to the device, the full matrices stay on the host.

    On a CUDA host the previous implementation handed a device tensor to the DataLoader, which is
    what runs out of GPU memory for large gene lists.

    :param monkeypatch: pytest monkeypatch fixture, used to capture the DataLoader input
    """
    captured: list[torch.Tensor] = []
    original_dataset = gee.DataSet

    def _capturing_dataset(data):
        captured.append(data)
        return original_dataset(data)

    monkeypatch.setattr(gee, "DataSet", _capturing_dataset)
    train = np.random.default_rng(0).normal(size=(8, _N_GENES))
    validation = np.random.default_rng(1).normal(size=(3, _N_GENES))

    train_gene_expession_autoencoder(train, validation, epochs_autoencoder=1)

    assert len(captured) == 1
    assert captured[0].device == _CPU
    assert captured[0].dtype == torch.float32


def test_training_runs_with_a_validation_set_smaller_than_the_batch() -> None:
    """Smoke test over the changed code path: float64 input, validation rows not a multiple of the batch."""
    train = np.random.default_rng(0).normal(size=(8, _N_GENES))
    validation = np.random.default_rng(1).normal(size=(3, _N_GENES))

    encoder = train_gene_expession_autoencoder(train, validation, epochs_autoencoder=2)

    assert isinstance(encoder, GeneExpressionEncoder)
    assert not encoder.training
    device = next(encoder.parameters()).device
    with torch.no_grad():
        assert encoder(torch.from_numpy(validation.astype(np.float32)).to(device)).shape == (3, encoder.latent_dim)
