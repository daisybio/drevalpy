"""Tests for the :mod:`drevalpy.registry.visualization` package surface (a facade over one registry singleton)."""

from __future__ import annotations

from drevalpy.registry import visualization
from drevalpy.registry.visualization._registry import visualization_registry
from tests._barrel_surface import SingletonFacadeSurface

FACADE_FUNCTIONS = ("applicable", "get", "list", "metadata", "register", "table")

EXPECTED_ORIGINS: dict[str, str] = {
    "VisualizationRegistry": "drevalpy.registry.visualization._registry",
    "visualization_registry": "drevalpy.registry.visualization._registry",
}


class TestVisualizationSurface(SingletonFacadeSurface):
    barrel = visualization
    origins = EXPECTED_ORIGINS
    unpinned_names = FACADE_FUNCTIONS
    callable_names = FACADE_FUNCTIONS
    singleton = visualization_registry
    keys_attribute = "names"
