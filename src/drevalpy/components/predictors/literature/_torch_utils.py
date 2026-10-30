"""Torch device selection shared by the literature predictors.

``torch`` is imported inside the functions: the predictors are registered on
``import drevalpy``, so the training stack must stay off that path (see
``tests/test_import_cost_policy.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import torch


def select_device() -> torch.device:
    """Return the CUDA device when one is available, else the CPU.

    :returns: The device to train and predict on.
    """
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DevicePicklingMixin:
    """Pickle a predictor without its torch device and re-select one when it is loaded.

    A fitted model trained on a GPU must still load on a CPU-only machine, so the device
    is not part of the pickle. After loading, the modules named in :attr:`device_modules`
    are moved to the device :func:`select_device` picks there; an attribute may hold one
    module, a list of modules, or ``None``.
    """

    #: Attributes holding the torch modules to move to the re-selected device.
    device_modules: ClassVar[tuple[str, ...]] = ("_model",)

    _device: Any

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
