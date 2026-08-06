"""Public drug response prediction models.

``construct_model`` returns thin generated ``DRPModel`` subclasses. Factory
dictionaries remain built-in-only compatibility views over the same path.
Every built-in factory name has a zoo YAML under ``drevalpy/models/zoo/``.
"""

from __future__ import annotations

from typing import Any

from drevalpy._deprecations import FACTORY_DICT_NAMES, warn_deprecated

from ._construct_model_api import build_builtin_factory_tables, construct_model
from ._model_persistence import load_model
from .drp_model import DRPModel

_FACTORY_MULTI, _FACTORY_SINGLE, _FACTORY_ALL = build_builtin_factory_tables()

__all__ = [
    "DRPModel",
    "MODEL_FACTORY",
    "MULTI_DRUG_MODEL_FACTORY",
    "SINGLE_DRUG_MODEL_FACTORY",
    "construct_model",
    "load_model",
]

_FACTORY_PUBLIC_TO_PRIVATE = {
    "MULTI_DRUG_MODEL_FACTORY": "_FACTORY_MULTI",
    "SINGLE_DRUG_MODEL_FACTORY": "_FACTORY_SINGLE",
    "MODEL_FACTORY": "_FACTORY_ALL",
}


def __getattr__(name: str) -> Any:
    if name in FACTORY_DICT_NAMES:
        warn_deprecated(
            what=name,
            replacement=('construct_model("ModelName"), config.from_spec("ModelName"), or list_zoo_names(scope=...)'),
            stacklevel=2,
        )
        value = globals()[_FACTORY_PUBLIC_TO_PRIVATE[name]]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
