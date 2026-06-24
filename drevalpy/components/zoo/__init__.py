"""Compatibility re-export — implementation lives in :mod:`drevalpy.models.zoo`."""

from drevalpy.models.zoo import (
    get_zoo_config,
    list_zoo_names,
    load_external_zoo_file,
    register_external_zoo_entry,
    zoo_model_config,
)

__all__ = [
    "get_zoo_config",
    "list_zoo_names",
    "load_external_zoo_file",
    "register_external_zoo_entry",
    "zoo_model_config",
]
