"""Tests for the :mod:`drevalpy.data.splitters` package surface.

The barrel re-exports only ``observed_mask``; the splitters are imported from their modules and
resolved through the registry, whose behaviour is tested in ``tests/registry/splitter/``. The
assertions themselves live in ``tests/_barrel_surface.py``.
"""

from __future__ import annotations

from drevalpy.data import splitters
from tests._barrel_surface import DeclaredSurface

#: ``exported name -> module that defines it``.
EXPECTED_ORIGINS: dict[str, str] = {
    "observed_mask": "drevalpy.data.splitters._folds",
}


class TestSplittersSurface(DeclaredSurface):
    barrel = splitters
    origins = EXPECTED_ORIGINS
