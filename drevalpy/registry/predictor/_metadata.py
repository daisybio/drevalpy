"""Build discovery/catalog metadata dicts for registered predictors."""

from __future__ import annotations

from typing import Any

from drevalpy.registry.components._metadata import base_component_metadata


def predictor_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, Any]:
    """Like `base_component_metadata` plus predictor input interface.

    :param registry_name: registry name.
    :param name: name.
    :param cls: Registered predictor class.
    :returns: Catalog metadata dict.
    """
    meta = base_component_metadata(registry_name, name, cls)
    meta["input_interface"] = getattr(cls, "input_interface", "")
    return meta
