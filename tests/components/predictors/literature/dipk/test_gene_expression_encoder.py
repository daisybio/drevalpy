"""Tests for the DIPK gene-expression autoencoder helpers."""

from __future__ import annotations

import inspect
import logging
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from drevalpy.components.predictors.literature.dipk import gene_expression_encoder as gee
from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import (
    CollateFn,
    DataSet,
    GeneExpressionDecoder,
    GeneExpressionEncoder,
    encode_gene_expression,
    train_gene_expession_autoencoder,
)
from drevalpy.experiment._tracking import tracking_run
from tests._fake_wandb import install_fake_wandb

INPUT_DIM = 6
SMALL_HIDDEN = [8, 4]


def _matrix(n_rows: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_rows, INPUT_DIM)).astype(np.float32)


def test_encoder_and_decoder_shapes_and_relu_bottleneck() -> None:
    h_dims = [8, 4]
    encoder = GeneExpressionEncoder(INPUT_DIM, latent_dim=5, h_dims=h_dims)
    decoder = GeneExpressionDecoder(INPUT_DIM, latent_dim=5, h_dims=SMALL_HIDDEN)
    encoder.eval()
    decoder.eval()
    features = torch.randn(3, INPUT_DIM)

    with torch.no_grad():
        embedding = encoder(features)
        reconstruction = decoder(embedding)

    assert embedding.shape == (3, 5)
    assert (embedding >= 0).all()
    assert encoder.latent_dim == 5
    assert reconstruction.shape == features.shape
    assert len(encoder.encoder) == len(h_dims)
    assert encoder.bottleneck.in_features == h_dims[-1]
    assert h_dims == [8, 4]


def test_dataset_and_collate_fn_work_together_in_a_dataloader() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    dataset = DataSet(tensor)

    assert len(dataset) == 4
    torch.testing.assert_close(dataset[2], tensor[2])
    assert CollateFn()([torch.ones(INPUT_DIM), torch.zeros(INPUT_DIM)]).shape == (2, INPUT_DIM)
    batches = list(DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=CollateFn()))
    assert len(batches) == 2
    assert batches[0].shape == (2, 3)


def test_encode_gene_expression_keeps_shape_squeezes_a_vector_and_leaves_eval_mode() -> None:
    encoder = GeneExpressionEncoder(INPUT_DIM, latent_dim=4, h_dims=SMALL_HIDDEN)
    encoder.train()

    encoded = encode_gene_expression(_matrix(3), encoder)

    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (3, 4)
    assert encode_gene_expression(_matrix(2)[0], encoder).shape == (4,)
    assert encoder.training is False


def test_train_autoencoder_returns_an_eval_encoder_usable_for_encoding() -> None:
    train = _matrix(4, seed=3)

    encoder = train_gene_expession_autoencoder(train, _matrix(4, seed=4), epochs_autoencoder=2)
    encoded = encode_gene_expression(train, encoder)

    assert isinstance(encoder, GeneExpressionEncoder)
    assert encoder.training is False
    assert encoded.shape == (4, encoder.latent_dim)
    assert np.isfinite(encoded).all()


def test_train_autoencoder_logs_each_epoch_to_the_active_tracking_run(monkeypatch) -> None:
    fake = install_fake_wandb(monkeypatch)

    with tracking_run("proj", name="r"):
        train_gene_expession_autoencoder(_matrix(4, seed=1), _matrix(4, seed=2), epochs_autoencoder=2)

    logged = fake.runs[0].logged
    assert [step["autoencoder_epoch"] for step in logged] == [1, 2]
    assert all({"autoencoder_train_loss", "autoencoder_val_loss"} <= set(step) for step in logged)


class TestAutoencoderFinalWeightsContract:
    pytestmark = pytest.mark.slow

    def test_early_stopping_returns_the_last_epoch_instead_of_restoring_the_best(self, caplog) -> None:
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


_CPU = torch.device("cpu")


def test_training_keeps_the_full_matrices_in_host_memory(monkeypatch) -> None:
    captured: list[torch.Tensor] = []
    original_dataset = gee.DataSet

    def _capturing_dataset(data):
        captured.append(data)
        return original_dataset(data)

    monkeypatch.setattr(gee, "DataSet", _capturing_dataset)

    train_gene_expession_autoencoder(_matrix(8), _matrix(3, seed=1), epochs_autoencoder=1)

    assert len(captured) == 1
    assert captured[0].device == _CPU
    assert captured[0].dtype == torch.float32


def test_only_batch_sized_tensors_are_moved_to_the_device(monkeypatch) -> None:
    n_genes, batch_size = 13, 4
    moved_rows: list[int] = []
    original_to = torch.Tensor.to

    def _recording_to(self, *args, **kwargs):
        # Data tensors have n_genes columns; the network weights are told apart by their wide side.
        if self.ndim == 2 and self.shape[1] == n_genes and self.shape[0] != 2048:
            moved_rows.append(self.shape[0])
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", _recording_to)
    rng = np.random.default_rng(0)
    train = rng.normal(size=(20, n_genes)).astype(np.float32)
    validation = rng.normal(size=(9, n_genes)).astype(np.float32)

    train_gene_expession_autoencoder(train, validation, epochs_autoencoder=2, batch_size=batch_size)

    assert moved_rows
    assert max(moved_rows) <= batch_size


def test_training_accepts_float64_input_and_a_validation_set_smaller_than_the_batch() -> None:
    train = _matrix(8).astype(np.float64)

    encoder = train_gene_expession_autoencoder(train, _matrix(3, seed=1), epochs_autoencoder=2)

    assert not encoder.training
    assert encode_gene_expression(_matrix(3), encoder).shape == (3, encoder.latent_dim)


def test_trainer_defaults_are_keyword_only() -> None:
    parameters = inspect.signature(train_gene_expession_autoencoder).parameters

    assert (parameters["lr"].default, parameters["batch_size"].default, parameters["patience"].default) == (
        1e-4,
        1024,
        3,
    )
    assert all(parameters[name].kind is inspect.Parameter.KEYWORD_ONLY for name in ("lr", "batch_size", "patience"))


def test_learning_rate_and_batch_size_reach_the_optimizer_and_loader(monkeypatch) -> None:
    seen: dict = {}
    original_adam, original_loader = gee.optim.Adam, gee.DataLoader

    def _capturing_adam(params, lr):
        seen["lr"] = lr
        return original_adam(params, lr=lr)

    def _capturing_loader(dataset, batch_size, **kwargs):
        seen["batch_size"] = batch_size
        return original_loader(dataset, batch_size=batch_size, **kwargs)

    monkeypatch.setattr(gee.optim, "Adam", _capturing_adam)
    monkeypatch.setattr(gee, "DataLoader", _capturing_loader)

    train_gene_expession_autoencoder(_matrix(8), _matrix(4, seed=1), epochs_autoencoder=1, lr=5e-3, batch_size=4)

    assert seen == {"lr": 5e-3, "batch_size": 4}


class _ConstantEncoder(nn.Module):
    latent_dim = 4

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.zeros(batch.shape[0], self.latent_dim, device=batch.device)


class _ConstantDecoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))
        self.input_dim = input_dim

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.zeros(embedding.shape[0], self.input_dim, device=embedding.device)


def test_patience_controls_when_early_stopping_triggers(monkeypatch, caplog) -> None:
    monkeypatch.setattr(gee, "GeneExpressionEncoder", _ConstantEncoder)
    monkeypatch.setattr(gee, "GeneExpressionDecoder", _ConstantDecoder)

    for patience in (1, 3):
        caplog.clear()
        with caplog.at_level(logging.INFO):
            train_gene_expession_autoencoder(
                _matrix(8), _matrix(4, seed=1), epochs_autoencoder=20, patience=patience, batch_size=4
            )
        epochs = [message for message in caplog.messages if message.startswith("DIPK Autoenc. Epoch:")]
        # Epoch 0 improves on the initial infinity, every later epoch counts towards patience.
        assert len(epochs) == patience + 1
