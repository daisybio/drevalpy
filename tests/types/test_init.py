"""Tests for the public :mod:`drevalpy.types` package surface.

:mod:`drevalpy.types` holds the types user code imports, so its ``__all__`` is a
compatibility promise. Only the re-export surface is pinned here; each type's
behaviour is tested beside its defining module.

Origins are recorded against the leaf module that defines each type. The four
assertions themselves live in ``tests/_barrel_surface.py``.
"""

from __future__ import annotations

from drevalpy import types
from tests._barrel_surface import DeclaredSurface

#: ``exported name -> module that defines it``.
EXPECTED_ORIGINS: dict[str, str] = {
    "Dataset": "drevalpy.types.data.dataset._dataset",
    "SplitMask": "drevalpy.types.data.split_mask",
    "SplitMasks": "drevalpy.types.data.split_masks",
}


class TestTypesSurface(DeclaredSurface):
    barrel = types
    origins = EXPECTED_ORIGINS
