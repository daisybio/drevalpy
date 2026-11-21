"""Tests for the top-level :mod:`drevalpy` package surface (origins recorded against sub-packages)."""

from __future__ import annotations

import importlib
import re
from types import ModuleType

import pytest

import drevalpy
from tests._barrel_surface import ReExportSurface

EXPECTED_ORIGINS: dict[str, str] = {
    "ModelSpec": "drevalpy.models",
    "DRPModel": "drevalpy.models",
    "load": "drevalpy.data.datasets",
    "randomization": "drevalpy.experiment",
    "robustness": "drevalpy.experiment",
    "run": "drevalpy._run",
    "single": "drevalpy._single",
    "split": "drevalpy.data",
}


class TestTopLevelSurface(ReExportSurface):
    barrel = drevalpy
    origins = EXPECTED_ORIGINS
    callable_names = tuple(EXPECTED_ORIGINS)


def test_the_registry_subpackage_is_bound_on_the_package() -> None:
    assert isinstance(drevalpy.registry, ModuleType)
    assert drevalpy.registry is importlib.import_module("drevalpy.registry")


def test_version_is_a_release_string() -> None:
    assert re.fullmatch(r"\d+\.\d+(\.\d+)?\S*", drevalpy.__version__), drevalpy.__version__


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        getattr(drevalpy, "definitely_not_a_real_symbol")  # noqa: B009
