"""Tests for the top-level :mod:`drevalpy` package surface.

``import drevalpy`` is the entry point every consumer and the CLI go through, and
this barrel re-exports the handful of names that make up the advertised API. It
carries no ``__all__``: the ``import x as x`` spelling is what marks each name as
re-exported for the type checker, so the promise is pinned by the explicit table
below, driven by ``tests/_barrel_surface.py``.

Origins are recorded against the sibling *sub-packages* rather than the private
modules behind them - that ``drevalpy.construct_model`` and
``drevalpy.models.construct_model`` are one object is the promise; which module
implements it is not.

The import-cost property of this same barrel is guarded separately in
``tests/test_import_cost_policy.py``, so every assertion here is an attribute
lookup on the already-imported module.
"""

from __future__ import annotations

import importlib
import re
from types import ModuleType

import pytest

import drevalpy
from tests._barrel_surface import ReExportSurface

#: ``top-level name -> sub-package it comes from``.
EXPECTED_ORIGINS: dict[str, str] = {
    "construct_model": "drevalpy.models",
    "load": "drevalpy.data.datasets",
    "randomization": "drevalpy.experiment",
    "robustness": "drevalpy.experiment",
    "run": "drevalpy._run",
    "single": "drevalpy._single",
    "split": "drevalpy.data",
}

#: Sub-packages re-exported eagerly so a bare ``import drevalpy`` is enough.
PROMISED_SUBMODULES = ("registry",)


class TestTopLevelSurface(ReExportSurface):
    barrel = drevalpy
    origins = EXPECTED_ORIGINS
    callable_names = tuple(EXPECTED_ORIGINS)


@pytest.mark.parametrize("name", sorted(PROMISED_SUBMODULES))
def test_promised_submodule_is_bound_on_the_package(name: str) -> None:
    submodule = getattr(drevalpy, name)
    assert isinstance(submodule, ModuleType)
    assert submodule is importlib.import_module(f"drevalpy.{name}")


def test_version_is_a_release_string() -> None:
    assert re.fullmatch(r"\d+\.\d+(\.\d+)?\S*", drevalpy.__version__), drevalpy.__version__


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        getattr(drevalpy, "definitely_not_a_real_symbol")  # noqa: B009
