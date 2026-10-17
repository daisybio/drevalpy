"""Early-stopped Lightning fits shared by MOLIR and SuperFELTR.

``MOLIModel.fit`` and ``train_superfeltr_model`` assembled the same trainer by hand:
monitor the validation loss when there is a validation loader and the training loss
otherwise, early-stop on it, checkpoint the best epoch into a freshly randomised
subdirectory, and silence the progress bar. Only MOLIR's ``save_weights_only`` and
pinned single device differed, so those are fields rather than forks.

``pytorch_lightning`` is imported inside the entry point. The leading underscore
keeps the module out of ``registry/_builtins.py::_discover_modules``.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from upath import UPath

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader


@dataclass(frozen=True)
class LightningRun:
    """Trainer settings for one early-stopped Lightning fit."""

    max_epochs: int
    patience: int = 5
    checkpoint_dir: str | UPath = "checkpoints"
    wandb_project: str | None = None
    save_weights_only: bool = False
    #: Pinned by MOLIR, left to Lightning's auto-detection elsewhere.
    devices: int | str | None = None
    enable_model_summary: bool = True


def run_lightning_fit(
    model: pl.LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    run: LightningRun,
) -> pl.callbacks.ModelCheckpoint:
    """Fit *model* with early stopping and return the checkpoint callback.

    :param model: The Lightning module to train.
    :param train_loader: Training loader.
    :param val_loader: Validation loader, or ``None`` to monitor the training loss.
    :param run: Trainer settings.
    :returns: The checkpoint callback holding the best epoch's path.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar

    monitor = "train_loss" if val_loader is None else "val_loss"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=UPath(run.checkpoint_dir) / _versioned_name(),
        monitor=monitor,
        mode="min",
        save_top_k=1,
        save_weights_only=run.save_weights_only,
    )
    trainer_kwargs: dict[str, Any] = {"enable_model_summary": run.enable_model_summary}
    if run.devices is not None:
        trainer_kwargs["devices"] = run.devices

    trainer = pl.Trainer(
        max_epochs=run.max_epochs,
        logger=_loggers(run.wandb_project),
        callbacks=[
            EarlyStopping(monitor=monitor, mode="min", patience=run.patience),
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=0),
        ],
        **trainer_kwargs,
    )
    if val_loader is None:
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, train_loader, val_loader)
    return checkpoint_callback


def _loggers(wandb_project: str | None) -> list[Any] | bool:
    """Resolve the Lightning ``logger`` argument for *wandb_project*.

    :param wandb_project: Weights & Biases project name, or ``None``.
    :returns: A one-element logger list, or ``True`` for Lightning's default logger.
    """
    if wandb_project is None:
        return True
    from pytorch_lightning.loggers import WandbLogger

    return [WandbLogger(project=wandb_project, log_model=False)]


def _versioned_name() -> str:
    """Return a random subdirectory name, so concurrent fits cannot collide.

    :returns: A ``version-<hex>`` directory name.
    """
    return "version-" + "".join(secrets.choice("0123456789abcdef") for _ in range(20))
