"""Resolve and rebind literature engine classes."""

from __future__ import annotations

import importlib

from drevalpy.components.predictors.literature._engine_base import LiteratureEngineBase

ENGINE_MODULES: dict[str, str] = {
    "PrecilyModel": "drevalpy.components.predictors.literature.impl.precily.precily",
    "SRMF": "drevalpy.components.predictors.literature.impl.srmf.srmf",
    "MOLIR": "drevalpy.components.predictors.literature.impl.molir.molir",
    "SuperFELTR": "drevalpy.components.predictors.literature.impl.superfeltr.superfeltr",
    "PharmaFormerModel": "drevalpy.components.predictors.literature.impl.pharmaformer.pharmaformer",
    "DIPKModel": "drevalpy.components.predictors.literature.impl.dipk.dipk",
    "SparseGOModel": "drevalpy.components.predictors.literature.impl.sparsego.sparsego",
}

# Preload-state key for data-derived hyperparameter discoveries (e.g. SparseGO drug_dim).
DISCOVERED_HYPERPARAMETERS_KEY = "discovered_hyperparameters"


def resolve_engine_cls(class_name: str) -> type[LiteratureEngineBase]:
    """Import a literature engine class by its legacy class name."""
    module_path = ENGINE_MODULES.get(class_name)
    if module_path is None:
        msg = f"Unknown literature engine class: {class_name}"
        raise ValueError(msg)
    module = importlib.import_module(module_path)
    engine_cls = getattr(module, class_name, None)
    if engine_cls is None or not issubclass(engine_cls, LiteratureEngineBase):
        msg = f"Module {module_path!r} does not export {class_name}"
        raise ValueError(msg)
    return engine_cls


def rebind_engine_class(engine: LiteratureEngineBase, class_name: str) -> LiteratureEngineBase:
    """Ensure *engine*'s class object matches the currently imported module."""
    current_cls = resolve_engine_cls(class_name)
    if type(engine) is not current_cls and type(engine).__name__ == current_cls.__name__:
        engine.__class__ = current_cls
    return engine
