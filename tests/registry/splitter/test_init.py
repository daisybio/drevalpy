"""Tests for the :mod:`drevalpy.registry.splitter` package surface.

The barrel is more than a re-export list: ``register``/``get``/``list``/``table``/
``metadata`` are thin module-level facades over the ``splitter_registry``
singleton, and the twenty-odd call sites in the package use the facade rather
than the singleton. Only the surface is asserted - the registry's own behaviour
belongs to ``test_registry.py`` beside it - and nothing here registers a mode,
so the process-global registry is left exactly as it was found. The assertions
themselves, forwarding included, live in ``tests/_barrel_surface.py``.

``table()`` is deliberately not called: it materialises a ``pandas`` DataFrame,
and ``tests/test_import_cost_policy.py`` exists precisely because pandas is
expensive. Its presence and callability are what the barrel promises.
"""

from __future__ import annotations

from drevalpy.registry import splitter
from drevalpy.registry.splitter._registry import splitter_registry
from tests._barrel_surface import SingletonFacadeSurface

#: Facade functions the barrel defines itself, with no module of their own.
FACADE_FUNCTIONS = ("get", "list", "metadata", "register", "table")

#: ``re-exported name -> private module that defines it``.
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
