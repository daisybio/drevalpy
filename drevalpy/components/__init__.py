"""Composable model components: featurizers and predictors.

Public registration and discovery functions are re-exported here for convenience.
The canonical source is ``drevalpy.registry``.
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy re-exports from drevalpy.registry to avoid circular imports."""
    _mapping = {
        "register_builtin_components": ("drevalpy.registry._builtins", "register_builtin_components"),
        "load_extensions": ("drevalpy.registry._extensions", "load_extensions"),
        "register_cell_line_featurizer": ("drevalpy.registry.cell_line_featurizer", "register"),
        "register_drug_featurizer": ("drevalpy.registry.drug_featurizer", "register"),
        "register_predictor": ("drevalpy.registry.predictor", "register"),
    }
    if name == "list_predictor_metadata":
        from drevalpy.registry.predictor import predictor_registry

        def list_predictor_metadata(*, tag: str | None = None) -> list[dict]:
            return predictor_registry.list_metadata(tag=tag)

        return list_predictor_metadata

    if name in _mapping:
        import importlib

        module_path, attr = _mapping[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)

    raise AttributeError(f"module 'drevalpy.components' has no attribute {name!r}")
