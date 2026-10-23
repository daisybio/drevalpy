"""Tests for the :mod:`drevalpy.data.splitters` package surface.

The barrel re-exports the registry types and the five built-in splitters, and
binds ``get_splitter`` to the registry singleton. Only the surface is asserted:
the registry's behaviour is tested in ``tests/registry/splitter/`` and each
splitter beside its module (``test_lpo.py`` and its siblings). The assertions
themselves live in ``tests/_barrel_surface.py``.
"""

from __future__ import annotations

from drevalpy.data import splitters
from tests._barrel_surface import DeclaredSurface

#: ``get_splitter`` is a bound method of the singleton, so no module holds the
#: very object the barrel does; resolving and calling it is what is pinned.
BOUND_NAMES = ("get_splitter",)

#: ``exported name -> module that defines it``.
EXPECTED_ORIGINS: dict[str, str] = {
    "SplitValidationError": "drevalpy.registry.splitter._validation",
    "Splitter": "drevalpy.registry.splitter._registry",
    "SplitterRegistry": "drevalpy.registry.splitter._registry",
    "Validation": "drevalpy.registry.splitter._validation",
    "leave_drug_out": "drevalpy.data.splitters.ldo",
    "leave_pair_out": "drevalpy.data.splitters.lpo",
    "leave_sample_out": "drevalpy.data.splitters.lso",
    "leave_study_out": "drevalpy.data.splitters.lsto",
    "leave_tissue_out": "drevalpy.data.splitters.lto",
    "splitter_registry": "drevalpy.registry.splitter._registry",
}


class TestSplittersSurface(DeclaredSurface):
    barrel = splitters
    origins = EXPECTED_ORIGINS
    unpinned_names = BOUND_NAMES
    callable_names = BOUND_NAMES
