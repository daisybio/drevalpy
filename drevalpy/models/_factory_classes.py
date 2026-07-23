"""Generate root DRPModel facade classes from zoo presets."""

from __future__ import annotations

from typing import Any

from drevalpy.models._native_drp_model import NativeDRPModel, create_native_drp_class
from drevalpy.models.config import ModelScope
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.zoo import get_zoo_config, list_zoo_names

# Public Python symbol -> zoo/factory name when they differ.
SYMBOL_TO_FACTORY_NAME: dict[str, str] = {
    "ElasticNetModel": "ElasticNet",
    "LassoModel": "Lasso",
    "SVMRegressor": "SVR",
    "DIPKModel": "DIPK",
    "PharmaFormerModel": "PharmaFormer",
    "PrecilyModel": "Precily",
}

FACTORY_NAME_TO_SYMBOL: dict[str, str] = {factory: symbol for symbol, factory in SYMBOL_TO_FACTORY_NAME.items()}


def symbol_for_factory_name(factory_name: str) -> str:
    """Return the root-exported Python symbol for a factory name."""
    return FACTORY_NAME_TO_SYMBOL.get(factory_name, factory_name)


def factory_name_for_symbol(symbol: str) -> str:
    """Return the factory/zoo name for a root-exported Python symbol."""
    return SYMBOL_TO_FACTORY_NAME.get(symbol, symbol)


def _early_stopping_from_predictor(config) -> bool:
    """Derive class-level early_stopping from the zoo predictor capability.

    Import failures are not swallowed as ``False``; callers should only invoke
    this when the predictor can be resolved.
    """
    from drevalpy.components.registry import get_predictor

    predictor_cls = get_predictor(config.predictor.name)
    return bool(getattr(predictor_cls, "supports_early_stopping", False))


def create_factory_class(factory_name: str) -> type[NativeDRPModel]:
    """Create the canonical facade class for one zoo factory entry."""
    config = get_zoo_config(factory_name)
    return create_native_drp_class(
        factory_name,
        spec=factory_name,
        class_name=symbol_for_factory_name(factory_name),
        scope=config.scope,
        validate_spec=False,
        class_dict={"early_stopping": _early_stopping_from_predictor(config)},
    )


def build_factory_tables() -> tuple[
    dict[str, type[DRPModel]],
    dict[str, type[DRPModel]],
    dict[str, type[DRPModel]],
    dict[str, type[NativeDRPModel]],
]:
    """Build multi/single/model factories and symbol->class mappings from the zoo."""
    multi: dict[str, type[DRPModel]] = {}
    single: dict[str, type[DRPModel]] = {}
    symbols: dict[str, type[NativeDRPModel]] = {}

    for factory_name in list_zoo_names(include_external=False):
        config = get_zoo_config(factory_name)
        cls = create_factory_class(factory_name)
        symbols[symbol_for_factory_name(factory_name)] = cls
        if config.scope == ModelScope.SINGLE_DRUG:
            single[factory_name] = cls
        else:
            multi[factory_name] = cls

    factory = {**multi, **single}
    return multi, single, factory, symbols


def populate_public_model_namespace(namespace: dict[str, Any]) -> None:
    """Install generated named classes and private factory tables.

    Public factory dict names are installed lazily by ``drevalpy.models.__getattr__``
    so deprecated access can emit a warning even after named facades were imported.
    """
    multi, single, factory, symbols = build_factory_tables()
    namespace.update(symbols)
    namespace.update(
        {
            "_FACTORY_MULTI": multi,
            "_FACTORY_SINGLE": single,
            "_FACTORY_ALL": factory,
        }
    )
