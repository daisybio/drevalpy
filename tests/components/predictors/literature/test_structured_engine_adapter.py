"""Tests for structured literature engine adapter."""

from __future__ import annotations

import importlib
from pathlib import Path

from drevalpy.components.predictors.literature.structured_engine_adapter import (
    ENGINE_MODULES,
    resolve_engine_cls,
)


def test_engine_modules_map_to_impl_packages() -> None:
    assert "PrecilyModel" in ENGINE_MODULES
    assert all("literature.impl" in path for path in ENGINE_MODULES.values())


def test_structured_engine_adapter_avoids_models_package_imports() -> None:
    module = importlib.import_module("drevalpy.components.predictors.literature.structured_engine_adapter")
    source_path = module.__file__
    assert source_path is not None
    text = Path(source_path).read_text(encoding="utf-8")
    assert "drevalpy.models.DIPK" not in text
    assert "drevalpy.components.predictors.literature.impl" in text


def test_resolve_engine_cls_imports_srmf_engine() -> None:
    from drevalpy.components.predictors.literature._engine_base import LiteratureEngineBase

    engine_cls = resolve_engine_cls("SRMF")
    assert issubclass(engine_cls, LiteratureEngineBase)
