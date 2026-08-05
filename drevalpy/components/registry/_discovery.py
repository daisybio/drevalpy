"""Ensure built-in components are registered for discovery listings."""

from __future__ import annotations


def ensure_builtins_for_discovery() -> None:
    """Restore all built-in components so listing reflects the full catalog.

    Partial lazy imports can leave a registry non-empty but incomplete.
    """
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
