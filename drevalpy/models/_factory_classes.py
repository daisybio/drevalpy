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

# Declared from zoo/predictor capability metadata without importing engines.
_EARLY_STOPPING_FACTORY_NAMES = frozenset(
    {
        "DIPK",
        "MOLIR",
        "MultiViewNeuralNetwork",
        "PharmaFormer",
        "SimpleNeuralNetwork",
        "SuperFELTR",
    }
)


def symbol_for_factory_name(factory_name: str) -> str:
    """Return the root-exported Python symbol for a factory name."""
    return FACTORY_NAME_TO_SYMBOL.get(factory_name, factory_name)


def factory_name_for_symbol(symbol: str) -> str:
    """Return the factory/zoo name for a root-exported Python symbol."""
    return SYMBOL_TO_FACTORY_NAME.get(symbol, symbol)


def create_factory_class(factory_name: str) -> type[NativeDRPModel]:
    """Create the canonical facade class for one zoo factory entry."""
    config = get_zoo_config(factory_name)
    return create_native_drp_class(
        factory_name,
        spec=factory_name,
        class_name=symbol_for_factory_name(factory_name),
        scope=config.scope,
        validate_spec=False,
        class_dict={"early_stopping": factory_name in _EARLY_STOPPING_FACTORY_NAMES},
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
        if factory_name == "SparseGO":
            symbols["SparseGOModel"] = cls
        if config.scope == ModelScope.SINGLE_DRUG:
            single[factory_name] = cls
        else:
            multi[factory_name] = cls

    factory = {**multi, **single}
    return multi, single, factory, symbols


def populate_public_model_namespace(namespace: dict[str, Any]) -> None:
    """Install generated factory tables and named classes into a module namespace."""
    multi, single, factory, symbols = build_factory_tables()
    namespace.update(symbols)
    namespace.update(
        {
            "MULTI_DRUG_MODEL_FACTORY": multi,
            "SINGLE_DRUG_MODEL_FACTORY": single,
            "MODEL_FACTORY": factory,
        }
    )
