"""Tests for the :mod:`drevalpy.registry.splitter` package surface."""

from __future__ import annotations

from drevalpy.registry import splitter
from tests._barrel_surface import DeclaredSurface

EXPECTED_ORIGINS: dict[str, str] = {
    "SplitValidationError": "drevalpy.registry.splitter._validation",
    "Splitter": "drevalpy.registry.splitter._registry",
    "Validation": "drevalpy.registry.splitter._validation",
    "canonical": "drevalpy.registry.splitter._registry",
    "get": "drevalpy.registry.splitter._registry",
    "register": "drevalpy.registry.splitter._registry",
    "splitter_registry": "drevalpy.registry.splitter._registry",
}


class TestSplitterSurface(DeclaredSurface):
    barrel = splitter
    origins = EXPECTED_ORIGINS
    callable_names = ("canonical", "get", "register")
