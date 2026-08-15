"""Tests for the public :mod:`drevalpy.models` package surface.

Forty-odd modules import ``DRPModel``, ``construct_model`` and ``load_model``
from this barrel rather than from the modules that define them, which is the
whole point of the barrel: the construction and persistence internals behind
these three names are free to move. This file therefore records the names and
their kinds and deliberately leaves the origin table empty, saying nothing about
which module supplies them.
"""

from __future__ import annotations

import inspect

from drevalpy import models
from tests._barrel_surface import DeclaredSurface

#: The names the module docstring and ``__all__`` promise to callers.
PROMISED_EXPORTS = ("DRPModel", "construct_model", "load_model")


class TestModelsSurface(DeclaredSurface):
    barrel = models
    unpinned_names = PROMISED_EXPORTS
    callable_names = ("construct_model", "load_model")


def test_drp_model_is_exported_as_a_class() -> None:
    assert inspect.isclass(models.DRPModel)
