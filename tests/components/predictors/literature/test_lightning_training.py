"""Tests for the shared Lightning fit wrapper.

``literature/_lightning_training.py`` replaced the trainer MOLIR's ``MOLIModel.fit``
and SuperFELTR's ``train_superfeltr_model`` each assembled by hand. What it decides,
and therefore what is pinned here: the monitored metric follows whether a validation
loader was supplied, the checkpoint lands in a randomised subdirectory of the caller's
directory, and MOLIR's two divergences (``save_weights_only`` and a pinned single
device) are honoured as fields rather than forks.

Every test runs a one-epoch fit of a tiny module, so the whole file is extended tier.
"""

from __future__ import annotations

import numpy as np
import pytest
import pytorch_lightning as pl
from upath import UPath

from drevalpy.components.predictors.literature._lightning_training import (
    LightningRun,
    _loggers,
    _versioned_name,
    run_lightning_fit,
)
from drevalpy.components.predictors.literature._omics_loaders import OmicsSplit, make_omics_loaders

#: Every test fits a real Lightning trainer for one epoch.
pytestmark = pytest.mark.slow

N_ENTITIES = 4
N_PAIRS = 4
FEATURE_DIM = 2


@pytest.fixture(autouse=True)
def _trainer_logs_in_tmp_path(monkeypatch, tmp_path) -> None:
    """Keep Lightning's default CSV logger out of the repository root."""
    monkeypatch.chdir(tmp_path)


def _split() -> OmicsSplit:
    rng = np.random.default_rng(0)
    return OmicsSplit(
        gene_expression=rng.normal(size=(N_ENTITIES, FEATURE_DIM)).astype(np.float32),
        mutations=rng.normal(size=(N_ENTITIES, FEATURE_DIM)).astype(np.float32),
        copy_number=rng.normal(size=(N_ENTITIES, FEATURE_DIM)).astype(np.float32),
        response=np.linspace(0.0, 1.0, N_PAIRS, dtype=np.float32),
        pair_idx=np.arange(N_PAIRS, dtype=np.int64) % N_ENTITIES,
    )


def _loaders(*, with_validation: bool):
    return make_omics_loaders(_split(), _split() if with_validation else None, batch_size=2)


class _TinyModule(pl.LightningModule):
    """Logs both losses so either monitor resolves, and records which ran."""

    def __init__(self) -> None:
        import torch
        from torch import nn

        super().__init__()
        self.layer = nn.Linear(FEATURE_DIM, 1)
        self.loss = nn.MSELoss()
        self.validated = False
        self._torch = torch

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        expression, _, _, response = batch
        loss = self.loss(self.layer(expression), response)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        expression, _, _, response = batch
        loss = self.loss(self.layer(expression), response)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.validated = True
        return loss

    def configure_optimizers(self):
        return self._torch.optim.SGD(self.parameters(), lr=0.01)


def _run(tmp_path, **overrides) -> LightningRun:
    settings: dict[str, object] = {"max_epochs": 1, "patience": 1, "checkpoint_dir": tmp_path}
    settings.update(overrides)
    return LightningRun(**settings)  # type: ignore[arg-type]


def test_without_a_validation_loader_the_training_loss_is_monitored(tmp_path) -> None:
    train_loader, val_loader = _loaders(with_validation=False)

    checkpoint = run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path))

    assert isinstance(checkpoint, pl.callbacks.ModelCheckpoint)
    assert checkpoint.monitor == "train_loss"


def test_with_a_validation_loader_the_validation_loss_is_monitored(tmp_path) -> None:
    train_loader, val_loader = _loaders(with_validation=True)

    checkpoint = run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path))

    assert checkpoint.monitor == "val_loss"


def test_the_validation_loader_is_actually_consumed(tmp_path) -> None:
    """Passing ``val_loader`` must reach ``trainer.fit``, not just switch the monitor."""
    module = _TinyModule()
    train_loader, val_loader = _loaders(with_validation=True)

    run_lightning_fit(module, train_loader, val_loader, _run(tmp_path))

    assert module.validated is True


def test_no_validation_loop_runs_without_a_validation_loader(tmp_path) -> None:
    module = _TinyModule()
    train_loader, val_loader = _loaders(with_validation=False)

    run_lightning_fit(module, train_loader, val_loader, _run(tmp_path))

    assert module.validated is False


def test_the_checkpoint_lands_under_a_randomised_subdirectory(tmp_path) -> None:
    train_loader, val_loader = _loaders(with_validation=False)

    checkpoint = run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path / "runs"))

    assert checkpoint.best_model_path
    assert checkpoint.best_model_path.endswith(".ckpt")
    assert str(checkpoint.dirpath).startswith(str(tmp_path / "runs"))
    assert "version-" in str(checkpoint.dirpath)


def test_two_fits_into_one_directory_do_not_share_a_checkpoint_path(tmp_path) -> None:
    paths = set()
    for _ in range(2):
        train_loader, val_loader = _loaders(with_validation=False)
        checkpoint = run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path))
        paths.add(str(checkpoint.dirpath))

    assert len(paths) == 2


def test_only_the_best_epoch_is_kept(tmp_path) -> None:
    train_loader, val_loader = _loaders(with_validation=False)

    checkpoint = run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path, max_epochs=3))

    assert checkpoint.save_top_k == 1
    assert len(list(UPath(checkpoint.dirpath).glob("*.ckpt"))) == 1


def test_save_weights_only_produces_a_checkpoint_without_optimizer_state(tmp_path) -> None:
    """MOLIR sets this; SuperFELTR does not, so the default must stay ``False``."""
    from drevalpy.utils.torch_io import load_trusted_mapping

    train_loader, val_loader = _loaders(with_validation=False)

    checkpoint = run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path, save_weights_only=True))

    payload = load_trusted_mapping(checkpoint.best_model_path, map_location="cpu")
    assert "state_dict" in payload
    assert "optimizer_states" not in payload


def test_the_default_keeps_optimizer_state(tmp_path) -> None:
    from drevalpy.utils.torch_io import load_trusted_mapping

    train_loader, val_loader = _loaders(with_validation=False)

    checkpoint = run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path))

    payload = load_trusted_mapping(checkpoint.best_model_path, map_location="cpu")
    assert "optimizer_states" in payload


def test_an_explicit_device_count_is_honoured(tmp_path) -> None:
    """MOLIR pins ``devices=1``; leaving it ``None`` must not pass the argument at all."""
    train_loader, val_loader = _loaders(with_validation=False)

    checkpoint = run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path, devices=1))

    assert checkpoint.best_model_path


def test_the_progress_bar_is_silenced(tmp_path, capsys) -> None:
    train_loader, val_loader = _loaders(with_validation=False)

    run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path))

    assert "it/s" not in capsys.readouterr().err


class TestLoggerSelection:
    """``wandb_project`` is threaded through from the predictors' hyperparameters."""

    def test_no_project_leaves_lightning_its_default_logger(self) -> None:
        assert _loggers(None) is True

    def test_a_project_yields_a_single_wandb_logger(self) -> None:
        from pytorch_lightning.loggers import WandbLogger

        loggers = _loggers("drevalpy-test")

        assert isinstance(loggers, list)
        assert len(loggers) == 1
        assert isinstance(loggers[0], WandbLogger)


class TestVersionedName:
    def test_the_name_is_prefixed_and_fixed_width(self) -> None:
        name = _versioned_name()

        assert name.startswith("version-")
        assert len(name) == len("version-") + 20
        assert set(name.removeprefix("version-")) <= set("0123456789abcdef")

    def test_names_do_not_repeat(self) -> None:
        assert len({_versioned_name() for _ in range(50)}) == 50
