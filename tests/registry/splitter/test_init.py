"""Tests for the :mod:`drevalpy.registry.splitter` package surface (a facade over one registry singleton)."""

from __future__ import annotations

from drevalpy.registry import splitter
from drevalpy.registry.splitter._registry import splitter_registry
from tests._barrel_surface import SingletonFacadeSurface

FACADE_FUNCTIONS = ("get", "list", "metadata", "register", "table")

EXPECTED_ORIGINS: dict[str, str] = {
    "SplitValidationError": "drevalpy.registry.splitter._validation",
    "Splitter": "drevalpy.registry.splitter._registry",
    "SplitterRegistry": "drevalpy.registry.splitter._registry",
    "Validation": "drevalpy.registry.splitter._validation",
    "splitter_registry": "drevalpy.registry.splitter._registry",
}


class TestSplitterSurface(SingletonFacadeSurface):
    barrel = splitter
    origins = EXPECTED_ORIGINS
    unpinned_names = FACADE_FUNCTIONS
    callable_names = FACADE_FUNCTIONS
    singleton = splitter_registry
    keys_attribute = "modes"
