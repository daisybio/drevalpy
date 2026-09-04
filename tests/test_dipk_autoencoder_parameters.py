"""Tests for the configurable parameters of the DIPK gene expression autoencoder."""

import inspect
import os

import numpy as np
import torch
import yaml
from torch import nn

from drevalpy.models.DIPK import gene_expression_encoder as gee
from drevalpy.models.DIPK.dipk import DIPKModel
from drevalpy.models.DIPK.gene_expression_encoder import (
    GeneExpressionDecoder,
    GeneExpressionEncoder,
    train_gene_expession_autoencoder,
)

_N_GENES = 16


def _toy_matrices() -> tuple[np.ndarray, np.ndarray]:
    """
    Build a tiny training and validation matrix.

    :returns: training and validation gene expression
    """
    rng = np.random.default_rng(0)
    return rng.normal(size=(8, _N_GENES)), rng.normal(size=(4, _N_GENES))


def test_defaults_reproduce_the_previous_hard_coded_values() -> None:
    """The new parameters must not change how DIPK trains when they are not passed."""
    defaults = inspect.signature(train_gene_expession_autoencoder).parameters

    assert defaults["lr"].default == 1e-4
    assert defaults["patience"].default == 3
    assert defaults["batch_size"].default == 1024
    assert defaults["encoder_state"].default is None
    assert defaults["decoder_state"].default is None


def test_warm_start_uses_the_given_weights() -> None:
    """With epochs_autoencoder=0 the returned encoder has to be exactly the warm start encoder."""
    train, validation = _toy_matrices()
    torch.manual_seed(0)
    pretrained = GeneExpressionEncoder(_N_GENES)
    encoder_state = {key: value.clone() for key, value in pretrained.state_dict().items()}

    encoder = train_gene_expession_autoencoder(train, validation, epochs_autoencoder=0, encoder_state=encoder_state)

    trained_state = encoder.state_dict()
    assert set(trained_state) == set(encoder_state)
    for key, value in encoder_state.items():
        assert torch.equal(trained_state[key].cpu(), value)


def test_warm_start_of_encoder_and_decoder_is_accepted() -> None:
    """Both state dicts are optional and independent of each other."""
    train, validation = _toy_matrices()
    torch.manual_seed(0)
    encoder_state = GeneExpressionEncoder(_N_GENES).state_dict()
    decoder_state = GeneExpressionDecoder(_N_GENES).state_dict()

    encoder = train_gene_expession_autoencoder(
        train,
        validation,
        epochs_autoencoder=1,
        encoder_state=encoder_state,
        decoder_state=decoder_state,
        batch_size=4,
    )

    assert isinstance(encoder, GeneExpressionEncoder)


def test_learning_rate_and_batch_size_are_forwarded(monkeypatch) -> None:
    """The learning rate reaches the optimizer and batch_size the DataLoader, not the former local constants.

    :param monkeypatch: pytest monkeypatch fixture, used to capture the constructor arguments
    """
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
    train, validation = _toy_matrices()

    train_gene_expession_autoencoder(train, validation, epochs_autoencoder=1, lr=5e-3, batch_size=4)

    assert seen == {"lr": 5e-3, "batch_size": 4}


class _ConstantEncoder(nn.Module):
    """Encoder replacement whose output never changes, so the validation loss is constant."""

    latent_dim = 4

    def __init__(self, input_dim: int) -> None:
        """
        Build a single unused parameter, the optimizer needs a non-empty parameter list.

        :param input_dim: number of genes, unused
        """
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))
        self.input_dim = input_dim

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Map every row to the same zero embedding.

        :param batch: input batch
        :returns: zero embedding
        """
        return torch.zeros(batch.shape[0], self.latent_dim, device=batch.device)


class _ConstantDecoder(nn.Module):
    """Decoder replacement whose output never changes."""

    def __init__(self, input_dim: int) -> None:
        """
        Build a single unused parameter.

        :param input_dim: number of genes to reconstruct
        """
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))
        self.input_dim = input_dim

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct every row as zeros.

        :param embedding: encoder output
        :returns: zero reconstruction
        """
        return torch.zeros(embedding.shape[0], self.input_dim, device=embedding.device)


def _count_epochs(capsys) -> int:
    """
    Count the epoch lines the autoencoder printed.

    :param capsys: pytest capsys fixture
    :returns: number of epochs that were run
    """
    return sum(1 for line in capsys.readouterr().out.splitlines() if line.startswith("DIPK Autoenc. Epoch:"))


def test_patience_controls_when_early_stopping_triggers(monkeypatch, capsys) -> None:
    """A constant validation loss never improves, so training has to stop patience epochs later.

    :param monkeypatch: pytest monkeypatch fixture, used to install constant models
    :param capsys: pytest capsys fixture, used to count the epochs
    """
    monkeypatch.setattr(gee, "GeneExpressionEncoder", _ConstantEncoder)
    monkeypatch.setattr(gee, "GeneExpressionDecoder", _ConstantDecoder)
    train, validation = _toy_matrices()

    for patience in (1, 3):
        train_gene_expession_autoencoder(train, validation, epochs_autoencoder=20, patience=patience, batch_size=4)
        # Epoch 0 improves on the initial infinity, every later epoch counts towards patience.
        assert _count_epochs(capsys) == patience + 1


def _capture_hook_arguments(monkeypatch) -> tuple[dict, GeneExpressionEncoder]:
    """
    Replace the autoencoder training with a stub that records what the DIPK hook passed to it.

    :param monkeypatch: pytest monkeypatch fixture, used to replace the autoencoder training
    :returns: the dict the stub fills and the encoder it returns
    """
    seen: dict = {}
    # The hook is annotated to return an encoder, so the stand in has to be a real one.
    stub_encoder = GeneExpressionEncoder(_N_GENES)

    def _fake_training(train_matrix, val_matrix, **kwargs):
        seen.update(kwargs)
        seen["shapes"] = (train_matrix.shape, val_matrix.shape)
        return stub_encoder

    monkeypatch.setattr("drevalpy.models.DIPK.dipk.train_gene_expession_autoencoder", _fake_training)
    return seen, stub_encoder


def test_dipk_fit_gene_encoder_hook_passes_all_autoencoder_hyperparameters(monkeypatch) -> None:
    """train() goes through the overridable hook, which has to forward all four autoencoder parameters.

    :param monkeypatch: pytest monkeypatch fixture, used to replace the autoencoder training
    """
    seen, stub_encoder = _capture_hook_arguments(monkeypatch)
    model = DIPKModel()
    model.hyperparameters = {
        "epochs_autoencoder": 7,
        "lr_autoencoder": 5e-3,
        "patience_autoencoder": 11,
        "batch_size_autoencoder": 32,
    }
    train, validation = _toy_matrices()

    assert model._fit_gene_encoder(train, validation) is stub_encoder
    assert seen == {
        "epochs_autoencoder": 7,
        "lr": 5e-3,
        "patience": 11,
        "batch_size": 32,
        "shapes": (train.shape, validation.shape),
    }


def test_dipk_hook_falls_back_to_the_previous_hard_coded_values(monkeypatch) -> None:
    """Hyperparameter sets without the optional autoencoder keys must keep training as before.

    :param monkeypatch: pytest monkeypatch fixture, used to replace the autoencoder training
    """
    seen, _ = _capture_hook_arguments(monkeypatch)
    model = DIPKModel()
    model.hyperparameters = {"epochs_autoencoder": 7}
    train, validation = _toy_matrices()

    model._fit_gene_encoder(train, validation)

    assert seen["lr"] == 1e-4
    assert seen["patience"] == 3
    assert seen["batch_size"] == 1024


def test_dipk_hook_does_not_read_the_prediction_network_parameters(monkeypatch) -> None:
    """batch_size, lr and patience belong to the predictor and must not leak into the autoencoder.

    :param monkeypatch: pytest monkeypatch fixture, used to replace the autoencoder training
    """
    seen, _ = _capture_hook_arguments(monkeypatch)
    model = DIPKModel()
    # The values of the shipped hyperparameters.yaml, which differ from the autoencoder defaults.
    model.hyperparameters = {"epochs_autoencoder": 7, "batch_size": 64, "lr": 0.001, "patience": 10}
    train, validation = _toy_matrices()

    model._fit_gene_encoder(train, validation)

    assert seen["lr"] == 1e-4
    assert seen["patience"] == 3
    assert seen["batch_size"] == 1024


def test_shipped_hyperparameters_keep_the_autoencoder_defaults() -> None:
    """The tuning grid has to offer the autoencoder keys without changing what DIPK does today."""
    with open(os.path.join(os.path.dirname(gee.__file__), "hyperparameters.yaml")) as handle:
        hpams = yaml.safe_load(handle)["DIPK"]

    assert hpams["lr_autoencoder"] == [1e-4]
    assert hpams["patience_autoencoder"] == [3]
    assert hpams["batch_size_autoencoder"] == [1024]
    # Same names without the suffix configure the prediction network and are deliberately different.
    assert hpams["patience"] != hpams["patience_autoencoder"]
    assert hpams["batch_size"] != hpams["batch_size_autoencoder"]
