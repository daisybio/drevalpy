"""Checkpointed early-stopping loop shared by the hand-rolled torch training loops.

``pharmaformer`` and ``dipk`` do not use Lightning; both ran the same skeleton
around their epoch functions - track the best validation loss, write the best
weights to a freshly randomised checkpoint file, stop after *patience* epochs
without improvement, then reload the best weights. Only the per-epoch work and
whether progress is logged differed.

``torch`` is imported inside the entry point: both callers live in a ``predictor.py``
that ``drevalpy.registry`` imports on ``import drevalpy``. See
``tests/test_import_cost_policy.py``. The leading underscore keeps the module out of
``registry/_builtins.py::_discover_modules``.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from upath import UPath

from drevalpy._utils.filenames import safe_filename
from drevalpy._utils.torch_io import load_state_dict, save_torch_payload
from drevalpy.experiment._tracking import log_metrics

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import torch
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EarlyStoppingRun:
    """How long to train, how long to wait, and where the best weights go."""

    epochs: int
    patience: int
    checkpoint_dir: str | UPath
    #: Used both in the checkpoint filename and as the log prefix.
    model_name: str
    verbose: bool = False


def train_with_early_stopping(
    model: Any,
    run: EarlyStoppingRun,
    train_epoch: Callable[[], float],
    val_epoch: Callable[[], float],
    device: torch.device,
) -> None:
    """Run *train_epoch*/*val_epoch* until patience runs out, then reload the best weights.

    :param model: Torch module being trained; its ``state_dict`` is checkpointed.
    :param run: Epoch budget, patience, checkpoint location and logging.
    :param train_epoch: Runs one training epoch and returns its mean loss.
    :param val_epoch: Runs one validation epoch and returns its mean loss.
    :param device: Device the reloaded weights are mapped onto.
    """
    checkpoint_path = _prepare_checkpoint_path(run)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    _log(run, f"Training {run.model_name} model")
    for epoch in range(run.epochs):
        train_loss = train_epoch()
        _log(run, f"{run.model_name}: Epoch [{epoch + 1}/{run.epochs}] Training Loss: {train_loss:.4f}")
        val_loss = val_epoch()
        _log(run, f"{run.model_name}: Epoch [{epoch + 1}/{run.epochs}] Validation Loss: {val_loss:.4f}")
        log_metrics({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_torch_payload(model.state_dict(), checkpoint_path)
            _log(run, f"{run.model_name}: Saved best model at epoch {epoch + 1}")
            continue

        epochs_without_improvement += 1
        if epochs_without_improvement >= run.patience:
            _log(run, f"{run.model_name}: Early stopping triggered at epoch {epoch + 1}")
            break

    _log(run, f"{run.model_name}: Reloading the best model")
    model.load_state_dict(load_state_dict(checkpoint_path, map_location=device))
    model.to(device)


def run_epoch(
    model: torch.nn.Module,
    loader: Iterable[Any],
    batch_loss: Callable[[Any], torch.Tensor],
    optimizer: Optimizer | None,
) -> float:
    """Run one training or evaluation epoch over *loader*.

    With an *optimizer* the model is in train mode and each batch loss is back-propagated and stepped;
    without one the model is in eval mode and no gradients are recorded.

    :param model: Module to train or evaluate.
    :param loader: Iterable of batches, each passed unchanged to *batch_loss*.
    :param batch_loss: Computes the loss tensor of one batch, moving the batch to the device itself.
    :param optimizer: Optimizer to step after each batch, or ``None`` for an evaluation epoch.
    :returns: Mean batch loss, or ``0.0`` for an empty *loader*.
    """
    import torch

    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    with torch.enable_grad() if training else torch.no_grad():
        for batch in loader:
            loss = batch_loss(batch)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.detach().item()
            count += 1
    return total / max(count, 1)


def _prepare_checkpoint_path(run: EarlyStoppingRun) -> UPath:
    """Create the checkpoint directory and return a collision-free file path.

    :param run: The run whose checkpoint directory and model name to use.
    :returns: Path the best weights are written to.
    """
    directory = UPath(run.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    version = "version-" + "".join(secrets.choice("0123456789abcdef") for _ in range(20))
    return directory / f"{version}_best_{safe_filename(run.model_name)}_model.pth"


def _log(run: EarlyStoppingRun, message: str) -> None:
    """Log *message* at INFO level when the run is verbose.

    :param run: The run whose verbosity to honour.
    :param message: Line to log.
    """
    if run.verbose:
        logger.info(message)
