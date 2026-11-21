"""Torch accelerator detection, the one place that asks torch which device to use.

Every torch component picks its device here, so ``DREVALPY_FORCE_CPU`` (or patching
:func:`accelerator_available`) keeps a whole run on the CPU. The check is backend-agnostic:
``torch.accelerator`` covers CUDA, Apple MPS, XPU and the other torch backends.

``torch`` is imported inside the functions, so ``import drevalpy`` stays light.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from drevalpy.errors import DrEvalError
from drevalpy.types.enums.accelerator import Accelerator

if TYPE_CHECKING:
    import torch

FORCE_CPU_ENV_VAR = "DREVALPY_FORCE_CPU"

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})


def accelerator_available() -> bool:
    """Return whether torch finds an accelerator and the CPU is not forced.

    :returns: ``False`` when ``DREVALPY_FORCE_CPU`` is set to a truthy value (``1``, ``true``,
        ``yes`` or ``on``, case-insensitive), else ``torch.accelerator.is_available()``.
    """
    if os.environ.get(FORCE_CPU_ENV_VAR, "").strip().lower() in _TRUTHY_VALUES:
        return False
    import torch

    return torch.accelerator.is_available()


def select_device() -> torch.device:
    """Return the current torch accelerator when one is available, else the CPU.

    :returns: The device to train and predict on.
    """
    import torch

    if accelerator_available():
        device = torch.accelerator.current_accelerator()
        if device is not None:
            return device
    return torch.device("cpu")


def lightning_accelerator() -> str:
    """Return the Lightning ``accelerator`` argument matching :func:`select_device`.

    :returns: ``"auto"`` when an accelerator is available, else ``"cpu"``.
    """
    return "auto" if accelerator_available() else "cpu"


def ensure_available(level: Accelerator, model: str) -> None:
    """Raise when *model* requires an accelerator and none is available.

    :param level: The model's accelerator level.
    :param model: Model name, for the message.
    :raises DrEvalError: If *level* is ``REQUIRED`` and :func:`accelerator_available` is ``False``.
    """
    if level is Accelerator.REQUIRED and not accelerator_available():
        msg = (
            f"Model {model!r} requires an accelerator (a GPU, Apple MPS or another torch backend), "
            f"but torch finds none. Run it on a machine with one, and unset {FORCE_CPU_ENV_VAR}."
        )
        raise DrEvalError(msg)
