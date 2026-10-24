"""``check_plugin`` driven through stub entry points; real plugin metadata is absent in this environment."""

from __future__ import annotations

import importlib.metadata
import itertools
import sys
import types
from typing import ClassVar

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.sample.base import SampleFeaturizer
from drevalpy.testing import plugins as plugins_module
from drevalpy.testing.plugins import (
    ENTRY_POINT_GROUP,
    PluginCheckError,
    PluginReport,
    check_plugin,
)
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block

PLUGIN_NAME = "fake_plugin"

_REGISTRY_NAME_COUNTER = itertools.count()


@pytest.fixture
def fake_plugin_package(monkeypatch):
    """Install a throwaway package holding one registered featurizer under a unique name."""
    from drevalpy.registry import sample_featurizer
    from drevalpy.registry.sample_featurizer import sample_featurizer_registry

    registry_name = f"fakePluginFeaturizer{next(_REGISTRY_NAME_COUNTER)}"
    package = types.ModuleType(PLUGIN_NAME)
    components = types.ModuleType(f"{PLUGIN_NAME}.components")
    monkeypatch.setitem(sys.modules, PLUGIN_NAME, package)
    monkeypatch.setitem(sys.modules, f"{PLUGIN_NAME}.components", components)

    @sample_featurizer.register(
        registry_name,
        description="Registered by the fake plugin package.",
        contract=FeatureFormat.NUMERIC_MATRIX,
        tags=("fake",),
    )
    class FakeFeaturizer(SampleFeaturizer):
        entity_id_only: ClassVar[bool] = True
        output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("fake", FeatureFormat.NUMERIC_MATRIX),)

        def _fit(self, source, **kwargs):
            return self

        def _transform_features(self, source, entity_ids):
            values = np.zeros((len(entity_ids), 1), dtype=np.float32)
            return FeaturizerResult(entity_ids, {"fake": numeric_feature_block(values)}, ("fake",))

        @property
        def output_dim(self) -> int:
            return 1

    FakeFeaturizer.__module__ = f"{PLUGIN_NAME}.components"
    components.FakeFeaturizer = FakeFeaturizer
    try:
        yield registry_name
    finally:
        sample_featurizer_registry._store.pop(registry_name, None)


def _stub_entry_points(monkeypatch, *entry_points: importlib.metadata.EntryPoint) -> None:
    def fake_entry_points(*, group: str):
        return list(entry_points) if group == ENTRY_POINT_GROUP else []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)


def _entry_point(value: str, name: str = PLUGIN_NAME) -> importlib.metadata.EntryPoint:
    return importlib.metadata.EntryPoint(name=name, value=value, group=ENTRY_POINT_GROUP)


@pytest.mark.parametrize(
    ("declared", "match"),
    [
        ((), "No drevalpy.plugins entry point"),
        ((), "Declared: none"),
        ((_entry_point("other.components", name="other"),), r"Declared: \['other'\]"),
        ((_entry_point("no_such_package_at_all.components"),), "failed to import"),
    ],
)
def test_entry_point_failures_are_reported(monkeypatch, declared, match):
    _stub_entry_points(monkeypatch, *declared)
    with pytest.raises(PluginCheckError, match=match):
        check_plugin(PLUGIN_NAME)


def test_discovery_failure_beats_a_reimport_and_clean_load_is_accepted(monkeypatch, fake_plugin_package):
    _stub_entry_points(monkeypatch, _entry_point(f"{PLUGIN_NAME}.components"))
    monkeypatch.setattr(plugins_module, "get_failed_plugins", lambda: {PLUGIN_NAME: "Traceback: boom"})
    with pytest.raises(PluginCheckError, match="failed to load during discovery"):
        check_plugin(PLUGIN_NAME)

    monkeypatch.setattr(plugins_module, "get_failed_plugins", dict)
    assert check_plugin(PLUGIN_NAME).name == PLUGIN_NAME


def test_a_plugin_registering_nothing_names_the_registries_it_checked(monkeypatch):
    monkeypatch.setitem(sys.modules, "empty_fake_plugin", types.ModuleType("empty_fake_plugin"))
    _stub_entry_points(monkeypatch, _entry_point("empty_fake_plugin", name="empty_fake_plugin"))
    with pytest.raises(PluginCheckError, match="registered nothing.*sample_featurizer"):
        check_plugin("empty_fake_plugin")


def test_successful_check_report(monkeypatch, fake_plugin_package):
    _stub_entry_points(monkeypatch, _entry_point(f"{PLUGIN_NAME}.components"))
    report = check_plugin(PLUGIN_NAME)

    assert isinstance(report, PluginReport)
    assert (report.name, report.value, report.module) == (PLUGIN_NAME, *[f"{PLUGIN_NAME}.components"] * 2)
    assert report.components["sample_featurizer"] == (fake_plugin_package,)
    assert "predictor" not in report.components
    assert "splitter" not in report.components
    assert report.component_names == (fake_plugin_package,)
    with pytest.raises(AttributeError):
        report.name = "other"  # type: ignore[misc]


def test_plugin_check_error_is_an_assertion_error():
    assert issubclass(PluginCheckError, AssertionError)


@pytest.mark.parametrize(
    ("registry_name", "component_name"),
    [
        ("sample_featurizer", "identity"),
        ("drug_featurizer", "fingerprints"),
        ("predictor", "ridge"),
        ("splitter", "LPO"),
        ("visualization", "critical_difference"),
    ],
)
def test_builtins_are_attributed_to_the_drevalpy_package(registry_name, component_name):
    resolved = plugins_module._REGISTRIES[registry_name].get(component_name)
    assert resolved.__module__.split(".")[0] == "drevalpy"


def test_all_five_registries_are_inspected():
    assert set(plugins_module._REGISTRIES) == {
        "sample_featurizer",
        "drug_featurizer",
        "predictor",
        "splitter",
        "visualization",
    }
