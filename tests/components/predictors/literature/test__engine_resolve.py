"""Tests for literature engine class resolution."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from drevalpy.components.predictors.literature._engine_base import LiteratureEngineBase
from drevalpy.components.predictors.literature._engine_resolve import ENGINE_MODULES, resolve_engine_cls


def test_engine_modules_map_to_impl_packages() -> None:
    assert "PrecilyModel" in ENGINE_MODULES
    assert "SRMF" in ENGINE_MODULES
    assert all("literature.impl" in path for path in ENGINE_MODULES.values())


def test_resolve_avoids_models_package_imports() -> None:
    module = importlib.import_module("drevalpy.components.predictors.literature._engine_resolve")
    source_path = module.__file__
    assert source_path is not None
    text = Path(source_path).read_text(encoding="utf-8")
    assert "drevalpy.models.DIPK" not in text
    assert "drevalpy.components.predictors.literature.impl" in text


def test_resolve_engine_cls_imports_srmf_engine() -> None:
    engine_cls = resolve_engine_cls("SRMF")
    assert issubclass(engine_cls, LiteratureEngineBase)


def test_resolve_engine_cls_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown literature engine class"):
        resolve_engine_cls("NotARealEngine")
