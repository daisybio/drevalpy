"""Training scaffolding shared by the literature predictors that run their own torch code.

* :func:`run_epoch` runs one training or evaluation epoch over a loader;
* :func:`train_with_early_stopping` is the checkpointed early-stopping loop around two epoch
  functions: track the best validation loss, write the best weights to a freshly randomised
  checkpoint file, stop after *patience* epochs without improvement, then reload the best weights;
* :class:`DevicePicklingMixin` selects the predictor's torch device and keeps it out of the pickle
  (see :func:`drevalpy._utils.accelerator.select_device`).

``pharmaformer``, ``paccmann`` and ``dipk`` use the early-stopping loop and none of them use Lightning;
``precily`` and ``sparsego`` run only :func:`run_epoch`.

``torch`` is imported inside the entry points: every caller lives in a ``predictor.py``
that ``drevalpy.registry`` imports on ``import drevalpy``. See
``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from upath import UPath

from drevalpy._utils.accelerator import select_device
from drevalpy._utils.filenames import safe_filename
from drevalpy._utils.torch_io import load_state_dict, save_torch_payload
from drevalpy.components.predictors._lightning_training import _versioned_name
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
    return directory / f"{_versioned_name()}_best_{safe_filename(run.model_name)}_model.pth"


def _log(run: EarlyStoppingRun, message: str) -> None:
    """Log *message* at INFO level when the run is verbose.

    :param run: The run whose verbosity to honour.
    :param message: Line to log.
    """
    if run.verbose:
        logger.info(message)


class DevicePicklingMixin:
    """Select a torch device for a predictor, and pickle it without the device.

    The device is selected on construction. A fitted model trained on an accelerator must still
    load on a machine without one, so the device
    is not part of the pickle. After loading, the modules named in :attr:`device_modules`
    are moved to the device :func:`select_device` picks there; an attribute may hold one
    module, a list of modules, or ``None``.
    """

    #: Attributes holding the torch modules to move to the re-selected device.
    device_modules: ClassVar[tuple[str, ...]] = ("_model",)

    _device: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        r"""Initialize the predictor, then select its device.

        :param \*args: Positional arguments of the predictor's ``__init__``.
        :param \**kwargs: Keyword arguments of the predictor's ``__init__``.
        """
        super().__init__(*args, **kwargs)
        self._device = select_device()

    def __getstate__(self) -> dict[str, Any]:
        """Return the pickled state, without the device.

        :returns: Instance attributes minus ``_device``.
        """
        state = dict(self.__dict__)
        state.pop("_device", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the attributes, select a device and move the modules there.

        :param state: Mapping returned by :meth:`__getstate__`.
        """
        self.__dict__.update(state)
        self._device = select_device()
        for name in self.device_modules:
            value = getattr(self, name, None)
            modules = value if isinstance(value, list) else [value]
            for module in modules:
                if module is not None:
                    module.to(self._device)
