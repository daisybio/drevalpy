"""Public drug response prediction models.

``construct_model`` returns thin generated ``DRPModel`` subclasses. Factory
dictionaries remain lazy built-in-only compatibility views over the same path.
Every built-in factory name has a zoo YAML under ``drevalpy/models/zoo/``.

Imports from this package are lazy so ``from drevalpy.models.config import ...``
does not pull the full runtime stack (avoids circular imports with components).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from drevalpy._deprecations import FACTORY_DICT_NAMES, warn_deprecated

__all__ = [
    "DRPModel",
    "construct_model",
    "MULTI_DRUG_MODEL_FACTORY",
    "SINGLE_DRUG_MODEL_FACTORY",
    "MODEL_FACTORY",
]

_LAZY_LOADED = False

if TYPE_CHECKING:
    from .drp_model import DRPModel

    SINGLE_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]]
    MULTI_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]]
    MODEL_FACTORY: dict[str, type[DRPModel]]


def _lazy_load_factory_tables() -> None:
    global _LAZY_LOADED
    if _LAZY_LOADED:
        return
    from drevalpy.models._construct_model_api import build_builtin_factory_tables

    multi, single, factory = build_builtin_factory_tables()
    globals().update(
        {
            "_FACTORY_MULTI": multi,
            "_FACTORY_SINGLE": single,
            "_FACTORY_ALL": factory,
        }
    )
    _LAZY_LOADED = True


_FACTORY_PUBLIC_TO_PRIVATE = {
    "MULTI_DRUG_MODEL_FACTORY": "_FACTORY_MULTI",
    "SINGLE_DRUG_MODEL_FACTORY": "_FACTORY_SINGLE",
    "MODEL_FACTORY": "_FACTORY_ALL",
}


def __getattr__(name: str) -> Any:
    if name == "DRPModel":
        from .drp_model import DRPModel

        globals()["DRPModel"] = DRPModel
        return DRPModel
    if name == "construct_model":
        from ._construct_model_api import construct_model

        return construct_model
    if name in FACTORY_DICT_NAMES:
        warn_deprecated(
            what=name,
            replacement=(
                'construct_model("ModelName"), ModelConfig.from_spec("ModelName"), or list_zoo_names(scope=...)'
            ),
            stacklevel=2,
        )
        _lazy_load_factory_tables()
        value = globals()[_FACTORY_PUBLIC_TO_PRIVATE[name]]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
