"""Tests for the shared Lightning fit wrapper."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import pytorch_lightning as pl
from upath import UPath

from drevalpy.components.predictors._lightning_training import (
    LightningRun,
    _loggers,
    _versioned_name,
    run_lightning_fit,
)
from drevalpy.components.predictors.literature._omics_loaders import OmicsSplit, make_omics_loaders
from drevalpy.experiment._tracking import tracking_run
from tests._fake_wandb import install_fake_wandb

pytestmark = pytest.mark.slow

N_ENTITIES = 4
N_PAIRS = 4
FEATURE_DIM = 2


@pytest.fixture(autouse=True)
def _trainer_logs_in_tmp_path(monkeypatch, tmp_path) -> None:
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


@pytest.mark.parametrize(("with_validation", "monitor"), [(False, "train_loss"), (True, "val_loss")])
def test_the_monitored_metric_follows_whether_a_validation_loader_is_supplied(
    tmp_path, with_validation: bool, monitor: str
) -> None:
    module = _TinyModule()
    train_loader, val_loader = _loaders(with_validation=with_validation)

    checkpoint = run_lightning_fit(module, train_loader, val_loader, _run(tmp_path))

    assert isinstance(checkpoint, pl.callbacks.ModelCheckpoint)
    assert checkpoint.monitor == monitor
    assert module.validated is with_validation


def test_checkpoints_land_in_distinct_randomised_subdirectories_and_only_the_best_is_kept(tmp_path) -> None:
    from drevalpy.utils.torch_io import load_trusted_mapping

    train_loader, val_loader = _loaders(with_validation=False)
    dirs = set()
    for _ in range(2):
        checkpoint = run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path / "runs", max_epochs=3))
        dirs.add(str(checkpoint.dirpath))

    assert len(dirs) == 2
    assert checkpoint.best_model_path.endswith(".ckpt")
    assert str(checkpoint.dirpath).startswith(str(tmp_path / "runs"))
    assert "version-" in str(checkpoint.dirpath)
    assert checkpoint.save_top_k == 1
    assert len(list(UPath(checkpoint.dirpath).glob("*.ckpt"))) == 1
    assert "optimizer_states" in load_trusted_mapping(checkpoint.best_model_path, map_location="cpu")


def test_save_weights_only_produces_a_checkpoint_without_optimizer_state(tmp_path) -> None:
    from drevalpy.utils.torch_io import load_trusted_mapping

    train_loader, val_loader = _loaders(with_validation=False)

    checkpoint = run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path, save_weights_only=True))

    payload = load_trusted_mapping(checkpoint.best_model_path, map_location="cpu")
    assert "state_dict" in payload
    assert "optimizer_states" not in payload


def _spy_trainer(monkeypatch) -> list[dict[str, object]]:
    seen: list[dict[str, object]] = []
    real_trainer = pl.Trainer

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real_trainer(*args, **kwargs)

    monkeypatch.setattr(pl, "Trainer", spy)
    return seen


def test_trainer_settings_follow_the_run(tmp_path, monkeypatch) -> None:
    seen = _spy_trainer(monkeypatch)
    train_loader, val_loader = _loaders(with_validation=True)
    run = _run(tmp_path, accelerator="cpu", devices=1, precision=32, log_every_n_steps=1, logger=False, patience=7)

    run_lightning_fit(_TinyModule(), train_loader, val_loader, run)
    run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path, enable_progress_bar=True))
    run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path, default_logger=False))

    first, progress, no_default_logger = seen
    assert (first["accelerator"], first["devices"], first["precision"]) == ("cpu", 1, 32)
    assert first["log_every_n_steps"] == 1
    assert first["logger"] is False
    assert first["enable_progress_bar"] is False
    (stopping,) = [c for c in list(first["callbacks"]) if isinstance(c, pl.callbacks.EarlyStopping)]  # type: ignore[attr-defined,arg-type]
    assert (stopping.patience, stopping.monitor) == (7, "val_loss")
    assert progress["enable_progress_bar"] is True
    assert no_default_logger["logger"] is False


def test_without_checkpointing_no_callback_is_attached_and_none_is_returned(tmp_path, monkeypatch) -> None:
    seen = _spy_trainer(monkeypatch)
    train_loader, val_loader = _loaders(with_validation=True)

    checkpoint = run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path, checkpoint=False))

    assert checkpoint is None
    assert seen[0]["callbacks"] == []
    assert seen[0]["enable_checkpointing"] is False
    assert not list(UPath(tmp_path).rglob("*.ckpt"))


@pytest.fixture
def recorded_loggers(monkeypatch) -> list[dict[str, object]]:
    import pytorch_lightning.loggers as lightning_loggers

    created: list[dict[str, object]] = []

    class _RecordingLogger:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)

    monkeypatch.setattr(lightning_loggers, "WandbLogger", _RecordingLogger)
    return created


def test_loggers_attach_a_wandb_logger_only_to_an_active_tracking_run(monkeypatch, recorded_loggers) -> None:
    fake = install_fake_wandb(monkeypatch)
    assert _loggers() is True
    assert _loggers(default=False) is False
    assert recorded_loggers == []

    with tracking_run("proj", name="r"):
        loggers = _loggers()
        kept = _loggers(default=False)

    assert isinstance(loggers, list)
    assert len(loggers) == 1
    assert isinstance(kept, list)
    assert len(kept) == 1
    assert recorded_loggers[0] == {"experiment": fake.runs[0], "log_model": False}


def test_wandb_never_imported_leaves_lightning_its_default_logger(monkeypatch, recorded_loggers) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)

    assert _loggers() is True


def test_a_fit_in_a_tracking_run_trains_with_the_wandb_logger(tmp_path, monkeypatch) -> None:
    import pytorch_lightning.loggers as lightning_loggers

    seen: list[object] = []
    real_trainer = pl.Trainer

    def spy_trainer(*args, **kwargs):
        seen.append(kwargs["logger"])
        kwargs["logger"] = False
        return real_trainer(*args, **kwargs)

    monkeypatch.setattr(lightning_loggers, "WandbLogger", lambda **kwargs: ("logger", kwargs["experiment"]))
    monkeypatch.setattr(pl, "Trainer", spy_trainer)
    fake = install_fake_wandb(monkeypatch)
    train_loader, val_loader = _loaders(with_validation=False)

    with tracking_run("proj", name="r"):
        run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path))
    run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path))

    assert seen == [[("logger", fake.runs[0])], True]


def test_versioned_names_are_prefixed_fixed_width_and_unique() -> None:
    name = _versioned_name()

    assert name.startswith("version-")
    assert len(name) == len("version-") + 20
    assert set(name.removeprefix("version-")) <= set("0123456789abcdef")
    assert len({_versioned_name() for _ in range(50)}) == 50
