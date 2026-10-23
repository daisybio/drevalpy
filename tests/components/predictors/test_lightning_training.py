"""Tests for the shared Lightning fit wrapper.

``_lightning_training.py`` replaced the trainer that the neural network, DrugGNN, MOLIR
and SuperFELTR each assembled by hand. What it decides, and therefore what is pinned
here: the monitored metric follows whether a validation loader was supplied, the
checkpoint lands in a randomised subdirectory of the caller's directory, and the
callers' divergences (``save_weights_only``, pinned devices, accelerator, precision,
logger, progress bar, no checkpointing) are honoured as fields rather than forks.

Every test runs a one-epoch fit of a tiny module, so the whole file is extended tier.
"""

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


def _spy_trainer(monkeypatch) -> list[dict[str, object]]:
    """Record the keyword arguments of every ``pl.Trainer`` built, then build a real one."""
    seen: list[dict[str, object]] = []
    real_trainer = pl.Trainer

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real_trainer(*args, **kwargs)

    monkeypatch.setattr(pl, "Trainer", spy)
    return seen


def test_trainer_settings_follow_the_run(tmp_path, monkeypatch) -> None:
    seen = _spy_trainer(monkeypatch)
    train_loader, val_loader = _loaders(with_validation=False)
    run = _run(tmp_path, accelerator="cpu", devices=1, precision=32, log_every_n_steps=1, logger=False)

    run_lightning_fit(_TinyModule(), train_loader, val_loader, run)

    (kwargs,) = seen
    assert kwargs["accelerator"] == "cpu"
    assert kwargs["devices"] == 1
    assert kwargs["precision"] == 32
    assert kwargs["log_every_n_steps"] == 1
    assert kwargs["logger"] is False
    assert kwargs["enable_progress_bar"] is False


def test_the_progress_bar_can_be_enabled(tmp_path, monkeypatch) -> None:
    seen = _spy_trainer(monkeypatch)
    train_loader, val_loader = _loaders(with_validation=False)

    run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path, enable_progress_bar=True))

    assert seen[0]["enable_progress_bar"] is True


def test_without_checkpointing_no_callback_is_attached_and_none_is_returned(tmp_path, monkeypatch) -> None:
    seen = _spy_trainer(monkeypatch)
    train_loader, val_loader = _loaders(with_validation=True)

    checkpoint = run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path, checkpoint=False))

    assert checkpoint is None
    assert seen[0]["callbacks"] == []
    assert seen[0]["enable_checkpointing"] is False
    assert not list(UPath(tmp_path).rglob("*.ckpt"))


def test_early_stopping_uses_the_requested_patience(tmp_path, monkeypatch) -> None:
    seen = _spy_trainer(monkeypatch)
    train_loader, val_loader = _loaders(with_validation=True)

    run_lightning_fit(_TinyModule(), train_loader, val_loader, _run(tmp_path, patience=7))

    callbacks: list[object] = list(seen[0]["callbacks"])  # type: ignore[call-overload]
    (stopping,) = [c for c in callbacks if isinstance(c, pl.callbacks.EarlyStopping)]
    assert stopping.patience == 7
    assert stopping.monitor == "val_loss"


class TestLoggerSelection:
    """A ``WandbLogger`` is attached to the active tracking run, and only then."""

    @pytest.fixture
    def recorded_loggers(self, monkeypatch) -> list[dict[str, object]]:
        """Replace ``WandbLogger`` with a recorder, so no real ``wandb`` run is needed."""
        import pytorch_lightning.loggers as lightning_loggers

        created: list[dict[str, object]] = []

        class _RecordingLogger:
            def __init__(self, **kwargs) -> None:
                created.append(kwargs)

        monkeypatch.setattr(lightning_loggers, "WandbLogger", _RecordingLogger)
        return created

    def test_no_run_leaves_lightning_its_default_logger(self, monkeypatch, recorded_loggers) -> None:
        install_fake_wandb(monkeypatch)

        assert _loggers() is True
        assert recorded_loggers == []

    def test_wandb_never_imported_leaves_lightning_its_default_logger(self, monkeypatch, recorded_loggers) -> None:
        monkeypatch.delitem(sys.modules, "wandb", raising=False)

        assert _loggers() is True

    def test_an_active_run_yields_a_single_logger_on_that_run(self, monkeypatch, recorded_loggers) -> None:
        fake = install_fake_wandb(monkeypatch)

        with tracking_run("proj", name="r"):
            loggers = _loggers()

        assert isinstance(loggers, list)
        assert len(loggers) == 1
        assert recorded_loggers == [{"experiment": fake.runs[0], "log_model": False}]

    def test_a_fit_in_a_tracking_run_trains_with_the_wandb_logger(self, tmp_path, monkeypatch) -> None:
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


class TestVersionedName:
    def test_the_name_is_prefixed_and_fixed_width(self) -> None:
        name = _versioned_name()

        assert name.startswith("version-")
        assert len(name) == len("version-") + 20
        assert set(name.removeprefix("version-")) <= set("0123456789abcdef")

    def test_names_do_not_repeat(self) -> None:
        assert len({_versioned_name() for _ in range(50)}) == 50
