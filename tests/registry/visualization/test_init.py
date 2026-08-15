"""Tests for the :mod:`drevalpy.registry.visualization` package surface.

Like its splitter sibling, this barrel is a facade over one process-global
registry singleton rather than a plain re-export list, and the sixteen dependents
in the package go through the facade. Every assertion is read-only, so no
registration is triggered and no registry teardown is needed; the assertions
themselves live in ``tests/_barrel_surface.py``.

``table()`` and ``applicable()`` are checked for presence only: the first builds
a ``pandas`` DataFrame and the second needs a full ``ExperimentResult``, and both
behaviours are owned by ``test_registry.py`` beside this file.
"""

from __future__ import annotations

from drevalpy.registry import visualization
from drevalpy.registry.visualization._registry import visualization_registry
from tests._barrel_surface import SingletonFacadeSurface

#: Facade functions the barrel defines itself, with no module of their own.
FACADE_FUNCTIONS = ("applicable", "get", "list", "metadata", "register", "table")

#: ``re-exported name -> private module that defines it``.
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
