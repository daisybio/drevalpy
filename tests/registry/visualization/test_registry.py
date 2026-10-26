"""Tests for :class:`~drevalpy.registry.visualization._registry.VisualizationRegistry`.

Every test that registers anything does so against a locally constructed registry;
the module singleton is only read.
"""

from __future__ import annotations

from typing import Any

import pytest

from drevalpy.errors import UnknownNameError
from drevalpy.registry import visualization as visualization_facade
from drevalpy.registry.visualization._registry import VisualizationRegistry, visualization_registry
from drevalpy.visualization.requirements import PlotRequirement
from tests.synthetic import make_experiment_result


def _cls(name: str) -> type:
    return type(name, (), {})


@pytest.fixture
def registry() -> VisualizationRegistry:
    return VisualizationRegistry()


@pytest.fixture
def violin(registry: VisualizationRegistry) -> type[Any]:
    return registry.register("violin", "distribution per model")(_cls("Violin"))


class TestRegistration:
    def test_register_stamps_the_name_and_lists_sorted_names(self, registry: VisualizationRegistry) -> None:
        assert registry.names == []

        zebra = registry.register("zebra")(_cls("Zebra"))
        registry.register("aardvark")(_cls("Aardvark"))

        assert zebra.registry_name == "zebra"
        assert registry.names == registry.list_names() == ["aardvark", "zebra"]
        assert registry.get("zebra") is zebra
        assert [row["name"] for row in registry.list_metadata()] == ["aardvark", "zebra"]

    def test_get_rejects_an_unknown_name(self, registry: VisualizationRegistry, violin: type[Any]) -> None:
        with pytest.raises(UnknownNameError, match=r"Unknown visualization 'heatmap'\. Registered: \['violin'\]"):
            registry.get("heatmap")
        with pytest.raises(ValueError, match="Unknown visualization 'heatmap'"):
            registry.get_metadata("heatmap")
        with pytest.raises(UnknownNameError, match="Unknown visualization 'heatmap'"):
            registry.result_type("heatmap")

    def test_describe(self, registry: VisualizationRegistry, violin: type[Any]) -> None:
        registry.register("bare")(_cls("Bare"))

        assert registry.describe("violin") == "distribution per model"
        assert registry.describe("bare") == ""
        assert registry.describe("heatmap") == ""

    def test_duplicate_registration_is_rejected_unless_overridden(
        self, registry: VisualizationRegistry, violin: type[Any]
    ) -> None:
        with pytest.raises(ValueError, match="Visualization 'violin' already registered"):
            registry.register("violin")(_cls("SecondViolin"))

        replacement = registry.register("violin", "replacement", override=True)(_cls("ReplacementViolin"))

        assert registry.get("violin") is replacement
        assert registry.describe("violin") == "replacement"

    def test_retain_only_drops_unlisted_names_and_frees_them(
        self, registry: VisualizationRegistry, violin: type[Any]
    ) -> None:
        registry.register("heatmap", "per-model heatmap")(_cls("Heatmap"))

        registry.retain_only(frozenset({"violin"}))
        assert registry.names == ["violin"]

        registry.retain_only(frozenset())
        assert registry.describe("violin") == ""
        replacement = registry.register("violin", "replacement")(_cls("ReplacementViolin"))
        assert registry.get("violin") is replacement

    def test_applicable_returns_the_classes_whose_requirements_the_experiment_meets(
        self, registry: VisualizationRegistry, violin: type[Any]
    ) -> None:
        folds = registry.register("folds", requirements=frozenset({PlotRequirement.MULTIPLE_FOLDS}))(_cls("Folds"))
        comparison = registry.register("comparison", requirements=frozenset({PlotRequirement.MULTIPLE_MODELS}))(
            _cls("Comparison")
        )

        assert registry.applicable(make_experiment_result(n_models=1, n_folds=1)) == [violin]
        assert registry.applicable(make_experiment_result(n_models=1, n_folds=2)) == [violin, folds]
        assert registry.applicable(make_experiment_result(n_models=2, n_folds=1)) == [violin, comparison]


class TestListing:
    def test_repr_and_dataframe(self, registry: VisualizationRegistry, violin: type[Any]) -> None:
        registry.register("folds", requirements=frozenset({PlotRequirement.MULTIPLE_FOLDS}))(_cls("Folds"))

        frame = registry.to_dataframe()
        rendered = repr(registry)

        assert list(frame.columns) == ["Name", "Description", "Result type", "Requirements"]
        assert frame.iloc[1].tolist()[:3] == ["violin", "distribution per model", "ExperimentResult"]
        assert str(PlotRequirement.MULTIPLE_FOLDS) in frame.iloc[0]["Requirements"]
        assert "distribution per model" in rendered
        assert not rendered.startswith("0")

    def test_repr_of_an_empty_registry_is_just_the_header(self, registry: VisualizationRegistry) -> None:
        assert repr(registry) == "Empty DataFrame\nColumns: []\nIndex: []"

    def test_metadata_and_result_type(self, registry: VisualizationRegistry, violin: type[Any]) -> None:
        registry.register("per_model", result_type="ModelResult")(_cls("PerModel"))

        assert registry.get_metadata("violin") == {
            "registry": "visualizations",
            "name": "violin",
            "class_name": "Violin",
            "description": "distribution per model",
            "result_type": "ExperimentResult",
            "requirements": frozenset(),
        }
        assert registry.get_metadata("per_model")["result_type"] == "ModelResult"
        assert registry.result_type("violin") == "ExperimentResult"
        assert registry.result_type("per_model") == "ModelResult"


class TestFacade:
    def test_the_singleton_is_populated_and_stamped_and_the_facade_delegates(self) -> None:
        name = visualization_registry.names[0]

        assert visualization_registry.get(name).registry_name == name
        assert visualization_facade.list() == visualization_registry.names
        assert visualization_facade.get(name) is visualization_registry.get(name)
        assert visualization_facade.metadata(name) == visualization_registry.get_metadata(name)
        assert visualization_facade.table()["Name"].tolist() == visualization_registry.names
        unconditional = [
            visualization_registry.get(name)
            for name in visualization_registry.names
            if not visualization_registry.get_metadata(name)["requirements"]
        ]
        assert visualization_facade.applicable(make_experiment_result(n_models=1, n_folds=1)) == unconditional

    def test_builtin_registration_is_a_no_op_once_populated(self) -> None:
        from drevalpy.registry._builtins import _register_builtin_visualizations

        before = visualization_registry.names

        _register_builtin_visualizations()

        assert visualization_registry.names == before

    def test_register_forwards_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded: dict[str, Any] = {}

        def fake_register(name, description, *, result_type, requirements, override):
            recorded["override"] = override
            return lambda cls: cls

        monkeypatch.setattr(visualization_registry, "register", fake_register)

        visualization_facade.register("x", "d", override=True)

        assert recorded["override"] is True
