"""Device handling shared by the literature predictors.

The device itself is chosen by :func:`drevalpy._utils.accelerator.select_device`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from drevalpy._utils.accelerator import select_device


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
