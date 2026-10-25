"""Early-stopped Lightning fits shared by the neural network, DrugGNN, MOLIR and SuperFELTR.

Each of them assembled the same trainer by hand: monitor the validation loss when there
is a validation loader and the training loss otherwise, early-stop on it, checkpoint the
best epoch into a freshly randomised subdirectory, and choose how loud the run is. The
few things that differ between them (``save_weights_only``, pinned devices, precision,
the accelerator, the progress bar, the logger, whether to checkpoint at all) are
fields of :class:`LightningRun` rather than forks.

``pytorch_lightning`` is imported inside the entry point. The leading underscore
keeps the module out of ``registry/_builtins.py::_discover_modules``.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from upath import UPath

from drevalpy.experiment._tracking import active_run

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader


@dataclass(frozen=True)
class LightningRun:
    """Trainer settings for one early-stopped Lightning fit.

    :param max_epochs: Upper bound on training epochs.
    :param checkpoint_dir: Directory that receives a randomised ``version-<hex>`` subdirectory.
    :param patience: Epochs without improvement before early stopping.
    :param save_weights_only: Store model weights only, without optimizer state.
    :param devices: Device count or spec, pinned by MOLIR and SuperFELTR and left to Lightning's auto-detection
        when ``None``.
    :param enable_model_summary: Print Lightning's model summary.
    :param accelerator: Lightning accelerator (``"cpu"`` pins the neural network to the CPU).
    :param precision: Numerical precision passed to the trainer.
    :param log_every_n_steps: Logging interval in training steps.
    :param logger: Attach the logger from :func:`_loggers`; ``False`` disables logging.
    :param default_logger: Keep Lightning's default logger when no W&B run is active. ``False`` logs to W&B
        only, so concurrent fits (one process per drug) do not race for the same ``lightning_logs`` version.
    :param enable_progress_bar: Show Lightning's progress bar.
    :param checkpoint: Early-stop on the monitored loss and keep the best epoch. When ``False`` neither
        callback is attached, which is the choice when there is no validation loader to monitor.
    """

    max_epochs: int
    checkpoint_dir: str | UPath
    patience: int = 5
    save_weights_only: bool = False
    devices: int | str | None = None
    enable_model_summary: bool = True
    accelerator: str = "auto"
    precision: int | str = 32
    log_every_n_steps: int = 50
    logger: bool = True
    default_logger: bool = True
    enable_progress_bar: bool = False
    checkpoint: bool = True


def run_lightning_fit(
    model: pl.LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    run: LightningRun,
) -> pl.callbacks.ModelCheckpoint | None:
    """Fit *model*, early-stopping on the validation loss (or the training loss without a loader).

    :param model: The Lightning module to train.
    :param train_loader: Training loader.
    :param val_loader: Validation loader, or ``None`` to monitor the training loss.
    :param run: Trainer settings.
    :returns: The checkpoint callback holding the best epoch's path, or ``None`` when ``run.checkpoint`` is
        ``False``.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping

    monitor = "train_loss" if val_loader is None else "val_loss"
    checkpoint_callback = None
    callbacks: list[pl.Callback] = []
    if run.checkpoint:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=UPath(run.checkpoint_dir) / _versioned_name(),
            monitor=monitor,
            mode="min",
            save_top_k=1,
            save_weights_only=run.save_weights_only,
        )
        callbacks = [EarlyStopping(monitor=monitor, mode="min", patience=run.patience), checkpoint_callback]

    trainer_kwargs: dict[str, Any] = {}
    if run.devices is not None:
        trainer_kwargs["devices"] = run.devices

    trainer = pl.Trainer(
        max_epochs=run.max_epochs,
        accelerator=run.accelerator,
        precision=run.precision,
        log_every_n_steps=run.log_every_n_steps,
        default_root_dir=str(run.checkpoint_dir),
        logger=_loggers(default=run.default_logger) if run.logger else False,
        callbacks=callbacks,
        enable_checkpointing=run.checkpoint,
        enable_model_summary=run.enable_model_summary,
        enable_progress_bar=run.enable_progress_bar,
        **trainer_kwargs,
    )
    if val_loader is None:
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, train_loader, val_loader)
    return checkpoint_callback


def _loggers(*, default: bool = True) -> list[Any] | bool:
    """Resolve the Lightning ``logger`` argument.

    :param default: Whether Lightning's default logger stands in when no W&B run is active.
    :returns: A one-element list logging into the active W&B tracking run, or ``default`` when no
        run is active.
    """
    wandb_run = active_run()
    if wandb_run is None:
        return default
    from pytorch_lightning.loggers import WandbLogger

    return [WandbLogger(experiment=wandb_run, log_model=False)]


def _versioned_name() -> str:
    """Return a random subdirectory name, so concurrent fits cannot collide.

    :returns: A ``version-<hex>`` directory name.
    """
    return "version-" + "".join(secrets.choice("0123456789abcdef") for _ in range(20))
