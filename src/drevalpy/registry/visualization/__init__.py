"""Visualization registry: register and retrieve plot classes, and select the applicable ones."""

from ._registry import applicable, register, visualization_registry

__all__ = ["applicable", "register", "visualization_registry"]
