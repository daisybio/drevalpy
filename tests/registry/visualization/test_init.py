"""Tests for the :mod:`drevalpy.registry.visualization` package surface."""

from __future__ import annotations

from drevalpy.registry import visualization
from tests._barrel_surface import DeclaredSurface

FUNCTIONS = ("applicable", "get", "register")


class TestVisualizationSurface(DeclaredSurface):
    barrel = visualization
    unpinned_names = (*FUNCTIONS, "visualization_registry")
    callable_names = FUNCTIONS
