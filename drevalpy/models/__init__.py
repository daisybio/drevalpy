"""Public drug response prediction models.

``construct_model`` returns thin generated ``DRPModel`` subclasses. Factory
dictionaries remain built-in-only compatibility views over the same path.
Every built-in factory name has a zoo YAML under ``drevalpy/models/zoo/``.
"""

from __future__ import annotations

import warnings
from typing import Any

from ._construct_model_api import build_builtin_factory_tables, construct_model
from ._model_persistence import load_model
from .drp_model import DRPModel

__all__ = [
    "DRPModel",
    "MODEL_FACTORY",
    "MULTI_DRUG_MODEL_FACTORY",
    "SINGLE_DRUG_MODEL_FACTORY",
    "construct_model",
    "load_model",
]

_FACTORY_DICT_NAMES = {"MODEL_FACTORY", "MULTI_DRUG_MODEL_FACTORY", "SINGLE_DRUG_MODEL_FACTORY"}

_FACTORY_PUBLIC_TO_PRIVATE = {
    "MULTI_DRUG_MODEL_FACTORY": "_FACTORY_MULTI",
    "SINGLE_DRUG_MODEL_FACTORY": "_FACTORY_SINGLE",
    "MODEL_FACTORY": "_FACTORY_ALL",
}

_FACTORY_MULTI: dict[str, type[Any]] | None = None
_FACTORY_SINGLE: dict[str, type[Any]] | None = None
_FACTORY_ALL: dict[str, type[Any]] | None = None


def _ensure_factory_tables() -> None:
    """Build built-in factory tables once, after package import has finished."""
    global _FACTORY_MULTI, _FACTORY_SINGLE, _FACTORY_ALL
    if _FACTORY_ALL is not None:
        return
    _FACTORY_MULTI, _FACTORY_SINGLE, _FACTORY_ALL = build_builtin_factory_tables()


def __getattr__(name: str) -> Any:
    if name in _FACTORY_DICT_NAMES:
        warnings.warn(
            f'{name} is deprecated; use construct_model("ModelName"), '
            'config.from_spec("ModelName"), or list_zoo_names(scope=...) instead.',
            FutureWarning,
            stacklevel=2,
        )
        _ensure_factory_tables()
        value = globals()[_FACTORY_PUBLIC_TO_PRIVATE[name]]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
