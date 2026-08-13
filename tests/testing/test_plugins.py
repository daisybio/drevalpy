"""Tests for :mod:`drevalpy.testing.plugins`.

``check_plugin`` inspects installed distribution metadata, and no plugin is
installed in drevalpy's own environment. The entry-point lookup is therefore
driven through a stub ``EntryPoint`` and a patched
``importlib.metadata.entry_points``, which is the same surface the real function
reads - the substitution is at the boundary, not inside the code under test.
"""

from __future__ import annotations

import importlib.metadata
import itertools
from typing import ClassVar

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.testing import plugins as plugins_module
from drevalpy.testing.plugins import (
    ENTRY_POINT_GROUP,
    PluginCheckError,
    PluginReport,
    check_plugin,
)
from drevalpy.types.data.batch.feature_block import BlockSpec, numeric_feature_block

PLUGIN_NAME = "fake_plugin"

_REGISTRY_NAME_COUNTER = itertools.count()


@pytest.fixture
def fake_plugin_package(monkeypatch):
    """Install a throwaway package holding one registered featurizer.

    The package is a real ``sys.modules`` entry rather than a file on disk: the
    entry point only needs an importable target, and ``check_plugin`` reads
    ``__module__`` to attribute components, so the root package name is all that
    actually has to be real.

    The registry name is unique per test because registration raises on a
    duplicate, and the entry is removed afterwards so the registry counts the
    policy tests assert stay intact.
    """
    import sys
    import types

    from drevalpy.registry import cell_line_featurizer
    from drevalpy.registry.cell_line_featurizer._registry import cell_line_featurizer_registry

    registry_name = f"fakePluginFeaturizer{next(_REGISTRY_NAME_COUNTER)}"
    package = types.ModuleType(PLUGIN_NAME)
    components = types.ModuleType(f"{PLUGIN_NAME}.components")
    monkeypatch.setitem(sys.modules, PLUGIN_NAME, package)
    monkeypatch.setitem(sys.modules, f"{PLUGIN_NAME}.components", components)

    @cell_line_featurizer.register(
        registry_name,
        description="Registered by the fake plugin package.",
        contract=FeatureFormat.NUMERIC_MATRIX,
        tags=("fake",),
    )
    class FakeFeaturizer(CellLineFeaturizer):
        """Fake featurizer contributed by the fake plugin."""

        entity_id_only: ClassVar[bool] = True
        output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("fake", FeatureFormat.NUMERIC_MATRIX),)

        def _fit(self, source, **kwargs):
            return self

        def _transform_blocks(self, source, entity_ids):
            return {"fake": numeric_feature_block(np.zeros((len(entity_ids), 1), dtype=np.float32))}

        @property
        def output_dim(self) -> int:
            return 1

    FakeFeaturizer.__module__ = f"{PLUGIN_NAME}.components"
    components.FakeFeaturizer = FakeFeaturizer
    try:
        yield registry_name
    finally:
        cell_line_featurizer_registry._store.pop(registry_name, None)


def _stub_entry_points(monkeypatch, *entry_points: importlib.metadata.EntryPoint) -> None:
    """Make ``entry_points(group=...)`` return exactly *entry_points*."""

    def fake_entry_points(*, group: str):
        return list(entry_points) if group == ENTRY_POINT_GROUP else []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)


def _entry_point(value: str, name: str = PLUGIN_NAME) -> importlib.metadata.EntryPoint:
    return importlib.metadata.EntryPoint(name=name, value=value, group=ENTRY_POINT_GROUP)


class TestUndeclaredEntryPoint:
    def test_a_missing_entry_point_is_reported(self, monkeypatch):
        _stub_entry_points(monkeypatch)

        with pytest.raises(PluginCheckError, match="No drevalpy.plugins entry point"):
            check_plugin(PLUGIN_NAME)

    def test_the_message_lists_what_is_declared(self, monkeypatch):
        _stub_entry_points(monkeypatch, _entry_point("other.components", name="other"))

        with pytest.raises(PluginCheckError, match=r"Declared: \['other'\]"):
            check_plugin(PLUGIN_NAME)

    def test_the_message_says_none_when_nothing_is_declared(self, monkeypatch):
        _stub_entry_points(monkeypatch)

        with pytest.raises(PluginCheckError, match="Declared: none"):
            check_plugin(PLUGIN_NAME)


class TestImportFailure:
    def test_an_unimportable_target_is_reported(self, monkeypatch):
        _stub_entry_points(monkeypatch, _entry_point("no_such_package_at_all.components"))

        with pytest.raises(PluginCheckError, match="failed to import"):
            check_plugin(PLUGIN_NAME)

    def test_a_failure_recorded_by_discovery_is_surfaced(self, monkeypatch, fake_plugin_package):
        """The loader's recorded traceback beats a re-import, which may now succeed."""
        _stub_entry_points(monkeypatch, _entry_point(f"{PLUGIN_NAME}.components"))
        monkeypatch.setattr(
            plugins_module,
            "get_failed_plugins",
            lambda: {PLUGIN_NAME: "Traceback (most recent call last): boom"},
        )

        with pytest.raises(PluginCheckError, match="failed to load during discovery"):
            check_plugin(PLUGIN_NAME)

    def test_a_clean_load_is_accepted(self, monkeypatch, fake_plugin_package):
        """Nothing recorded against the plugin means the check proceeds."""
        _stub_entry_points(monkeypatch, _entry_point(f"{PLUGIN_NAME}.components"))
        monkeypatch.setattr(plugins_module, "get_failed_plugins", dict)

        assert check_plugin(PLUGIN_NAME).name == PLUGIN_NAME


class TestNoComponentsRegistered:
    def test_a_plugin_registering_nothing_is_reported(self, monkeypatch):
        import sys
        import types

        monkeypatch.setitem(sys.modules, "empty_fake_plugin", types.ModuleType("empty_fake_plugin"))
        _stub_entry_points(monkeypatch, _entry_point("empty_fake_plugin", name="empty_fake_plugin"))

        with pytest.raises(PluginCheckError, match="registered nothing"):
            check_plugin("empty_fake_plugin")

    def test_the_message_names_the_registries_it_checked(self, monkeypatch):
        import sys
        import types

        monkeypatch.setitem(sys.modules, "empty_fake_plugin", types.ModuleType("empty_fake_plugin"))
        _stub_entry_points(monkeypatch, _entry_point("empty_fake_plugin", name="empty_fake_plugin"))

        with pytest.raises(PluginCheckError, match="cell_line_featurizer"):
            check_plugin("empty_fake_plugin")


class TestSuccessfulCheck:
    @pytest.fixture
    def report(self, monkeypatch, fake_plugin_package) -> PluginReport:
        _stub_entry_points(monkeypatch, _entry_point(f"{PLUGIN_NAME}.components"))
        return check_plugin(PLUGIN_NAME)

    def test_it_returns_a_report(self, report):
        assert isinstance(report, PluginReport)

    def test_the_report_names_the_plugin(self, report):
        assert report.name == PLUGIN_NAME

    def test_the_report_records_the_declared_value(self, report):
        assert report.value == f"{PLUGIN_NAME}.components"

    def test_the_report_records_the_resolved_module(self, report):
        assert report.module == f"{PLUGIN_NAME}.components"

    def test_the_contributed_component_is_attributed_to_its_registry(self, report, fake_plugin_package):
        assert report.components["cell_line_featurizer"] == (fake_plugin_package,)

    def test_registries_the_plugin_did_not_touch_are_omitted(self, report):
        assert "predictor" not in report.components
        assert "splitter" not in report.components

    def test_component_names_flattens_every_registry(self, report, fake_plugin_package):
        assert report.component_names == (fake_plugin_package,)

    def test_builtin_components_are_not_attributed_to_the_plugin(self, report):
        """Attribution is by root package, so ``identity`` must not appear."""
        assert "identity" not in report.component_names

    def test_the_report_is_immutable(self, report):
        with pytest.raises(AttributeError):
            report.name = "other"  # type: ignore[misc]


class TestErrorType:
    def test_plugin_check_error_is_an_assertion_error(self):
        """So a failure reads as a test failure, not an error, inside a test."""
        assert issubclass(PluginCheckError, AssertionError)


class TestAttributionAcrossRegistries:
    """Attribution reads ``__module__`` off whatever ``get`` returns.

    That is a class for four registries but a *function* for the splitter
    registry, and one wrapped in validation at that - so the wrapper must
    preserve ``__module__`` or a plugin's splitter would go unattributed.
    """

    @pytest.mark.parametrize(
        ("registry_name", "component_name", "expected_root"),
        [
            ("cell_line_featurizer", "identity", "drevalpy"),
            ("drug_featurizer", "fingerprints", "drevalpy"),
            ("predictor", "ridge", "drevalpy"),
            ("splitter", "LPO", "drevalpy"),
            ("visualization", "critical_difference", "drevalpy"),
        ],
    )
    def test_a_builtin_is_attributed_to_the_drevalpy_package(self, registry_name, component_name, expected_root):
        module = plugins_module._REGISTRIES[registry_name]

        resolved = module.get(component_name)

        assert resolved.__module__.split(".")[0] == expected_root

    def test_all_five_registries_are_inspected(self):
        assert set(plugins_module._REGISTRIES) == {
            "cell_line_featurizer",
            "drug_featurizer",
            "predictor",
            "splitter",
            "visualization",
        }
