"""Compatibility re-exports for DRPModel bridge helpers.

Prefer :mod:`drevalpy.models._component_bridge` for new model-side adapter code.
"""

from __future__ import annotations

from typing import Any

_BRIDGE_EXPORTS = frozenset(
    {
        "ComponentDRPBridge",
        "preview_sklearn_estimator",
        "restore_naive_to_components",
        "restore_sklearn_to_components",
        "sync_naive_from_components",
        "sync_sklearn_from_components",
    }
)

__all__ = [  # noqa: F822
    "ComponentDRPBridge",
    "ensure_components_registered",
    "preview_sklearn_estimator",
    "restore_naive_to_components",
    "restore_sklearn_to_components",
    "sync_naive_from_components",
    "sync_sklearn_from_components",
]


def ensure_components_registered(*args: Any, **kwargs: Any) -> None:
    from drevalpy.components.register_builtins import ensure_components_registered as _ensure

    _ensure(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name in _BRIDGE_EXPORTS:
        from drevalpy.models import _component_bridge

        return getattr(_component_bridge, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
