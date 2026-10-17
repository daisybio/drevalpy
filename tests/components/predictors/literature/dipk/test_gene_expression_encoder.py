"""Tests for the DIPK gene-expression autoencoder helpers.

The autoencoder trains on CPU here with a very small matrix and one or two
epochs; the point is to cover the training loop, the collate/dataset plumbing,
and the encode path rather than to reach a useful reconstruction.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import (
    CollateFn,
    DataSet,
    GeneExpressionDecoder,
    GeneExpressionEncoder,
    encode_gene_expression,
    train_gene_expession_autoencoder,
)

INPUT_DIM = 6
SMALL_HIDDEN = [8, 4]


def _matrix(n_rows: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_rows, INPUT_DIM)).astype(np.float32)


def test_encoder_projects_rows_to_the_latent_dim() -> None:
    encoder = GeneExpressionEncoder(INPUT_DIM, latent_dim=5, h_dims=SMALL_HIDDEN)
    encoder.eval()

    with torch.no_grad():
        embedding = encoder(torch.randn(3, INPUT_DIM))

    assert embedding.shape == (3, 5)
    assert encoder.latent_dim == 5


def test_encoder_output_is_non_negative_after_the_relu_bottleneck() -> None:
    encoder = GeneExpressionEncoder(INPUT_DIM, latent_dim=4, h_dims=SMALL_HIDDEN)
    encoder.eval()

    with torch.no_grad():
        embedding = encoder(torch.randn(3, INPUT_DIM))

    assert (embedding >= 0).all()


def test_encoder_builds_one_block_per_hidden_dim() -> None:
    encoder = GeneExpressionEncoder(INPUT_DIM, latent_dim=4, h_dims=SMALL_HIDDEN)

    assert len(encoder.encoder) == len(SMALL_HIDDEN)
    assert encoder.bottleneck.in_features == SMALL_HIDDEN[-1]


def test_encoder_does_not_mutate_the_hidden_dims_argument() -> None:
    h_dims = [8, 4]

    GeneExpressionEncoder(INPUT_DIM, latent_dim=4, h_dims=h_dims)

    assert h_dims == [8, 4]


def test_decoder_restores_the_input_width() -> None:
    decoder = GeneExpressionDecoder(INPUT_DIM, latent_dim=5, h_dims=SMALL_HIDDEN)
    decoder.eval()

    with torch.no_grad():
        reconstruction = decoder(torch.randn(3, 5))

    assert reconstruction.shape == (3, INPUT_DIM)


def test_encoder_and_decoder_compose_into_a_reconstruction() -> None:
    encoder = GeneExpressionEncoder(INPUT_DIM, latent_dim=4, h_dims=SMALL_HIDDEN)
    decoder = GeneExpressionDecoder(INPUT_DIM, latent_dim=4, h_dims=SMALL_HIDDEN)
    encoder.eval()
    decoder.eval()
    features = torch.randn(3, INPUT_DIM)

    with torch.no_grad():
        reconstruction = decoder(encoder(features))

    assert reconstruction.shape == features.shape


def test_collate_fn_stacks_row_tensors_into_a_batch() -> None:
    rows = [torch.ones(INPUT_DIM), torch.zeros(INPUT_DIM)]

    batch = CollateFn()(rows)

    assert batch.shape == (2, INPUT_DIM)


def test_dataset_reports_its_length_and_indexes_rows() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    dataset = DataSet(tensor)

    assert len(dataset) == 4
    torch.testing.assert_close(dataset[2], tensor[2])


def test_dataset_and_collate_fn_work_together_in_a_dataloader() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    loader = DataLoader(DataSet(tensor), batch_size=2, shuffle=False, collate_fn=CollateFn())
    batches = list(loader)

    assert len(batches) == 2
    assert batches[0].shape == (2, 3)


def test_encode_gene_expression_keeps_the_matrix_shape() -> None:
    encoder = GeneExpressionEncoder(INPUT_DIM, latent_dim=4, h_dims=SMALL_HIDDEN)

    encoded = encode_gene_expression(_matrix(3), encoder)

    assert encoded.shape == (3, 4)
    assert isinstance(encoded, np.ndarray)


def test_encode_gene_expression_squeezes_a_single_vector_back_to_one_dimension() -> None:
    encoder = GeneExpressionEncoder(INPUT_DIM, latent_dim=4, h_dims=SMALL_HIDDEN)
    encoder.eval()

    encoded = encode_gene_expression(_matrix(2)[0], encoder)

    assert encoded.shape == (4,)


def test_encode_gene_expression_leaves_the_encoder_in_eval_mode() -> None:
    encoder = GeneExpressionEncoder(INPUT_DIM, latent_dim=4, h_dims=SMALL_HIDDEN)
    encoder.train()

    encode_gene_expression(_matrix(3), encoder)

    assert encoder.training is False


def test_train_autoencoder_returns_an_encoder_in_eval_mode() -> None:
    train = _matrix(4, seed=1)
    validation = _matrix(4, seed=2)

    encoder = train_gene_expession_autoencoder(train, validation, epochs_autoencoder=1)

    assert isinstance(encoder, GeneExpressionEncoder)
    assert encoder.training is False


def test_train_autoencoder_produces_an_encoder_usable_for_encoding() -> None:
    train = _matrix(4, seed=3)
    validation = _matrix(4, seed=4)

    encoder = train_gene_expession_autoencoder(train, validation, epochs_autoencoder=2)
    encoded = encode_gene_expression(train, encoder)

    assert encoded.shape == (4, encoder.latent_dim)
    assert np.isfinite(encoded).all()


class TestAutoencoderFinalWeightsContract:
    pytestmark = pytest.mark.slow

    def test_early_stopping_returns_the_last_epoch_instead_of_restoring_the_best(self, caplog) -> None:
        """DIPK pretraining saves epoch 1000's final weights; local early stopping preserves that contract."""
        random.seed(2)
        np.random.seed(2)
        torch.manual_seed(2)
        rng = np.random.default_rng(1002)
        train = rng.normal(size=(4, INPUT_DIM)).astype(np.float32)
        validation = (rng.normal(size=(4, INPUT_DIM)) * 3 + 5).astype(np.float32)
        validation_snapshots: list[dict[str, torch.Tensor]] = []

        def capture_validation_state(module, _inputs, _output) -> None:
            if isinstance(module, GeneExpressionEncoder) and not module.training:
                validation_snapshots.append(
                    {name: value.detach().clone() for name, value in module.state_dict().items()}
                )

        handle = torch.nn.modules.module.register_module_forward_hook(capture_validation_state)
        try:
            with caplog.at_level(logging.INFO):
                encoder = train_gene_expession_autoencoder(train, validation, epochs_autoencoder=10)
        finally:
            handle.remove()

        assert "DIPK Autoenc. Early stopping triggered at epoch 3" in caplog.messages
        assert len(validation_snapshots) == 4
        returned_state = encoder.state_dict()
        for name, value in returned_state.items():
            torch.testing.assert_close(value, validation_snapshots[-1][name])
        assert any(
            not torch.equal(validation_snapshots[0][name], validation_snapshots[-1][name]) for name in returned_state
        )
