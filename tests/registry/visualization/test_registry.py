"""Tests for :class:`~drevalpy.registry.visualization._registry.VisualizationRegistry`.

Registration is destructive - ``register`` raises on a duplicate name - and the
autouse ``_ensure_registries_populated`` fixture in ``tests/conftest.py`` keeps the
module singleton populated for the whole session. So every test that registers
anything does so against a locally constructed ``VisualizationRegistry``, and the
singleton is only ever read.
"""

from __future__ import annotations

from typing import Any

import pytest

from drevalpy.registry import visualization as visualization_facade
from drevalpy.registry.visualization._registry import VisualizationRegistry, visualization_registry
from drevalpy.visualization.requirements import PlotRequirement


class _RecordingExperiment:
    """``ExperimentResult`` stand-in recording the requirement sets it was asked about."""

    def __init__(self, *, answer: bool = True) -> None:
        self.answer = answer
        self.asked: list[frozenset[Any]] = []

    def satisfies(self, requirements: frozenset[Any]) -> bool:
        """Record the query and return the canned answer."""
        self.asked.append(requirements)
        return self.answer


class _SelectiveExperiment:
    """``ExperimentResult`` stand-in that only satisfies single-model plots."""

    def satisfies(self, requirements: frozenset[Any]) -> bool:
        """Reject anything needing more than one model."""
        return PlotRequirement.MULTIPLE_MODELS not in requirements


@pytest.fixture
def registry() -> VisualizationRegistry:
    return VisualizationRegistry()


@pytest.fixture
def violin(registry: VisualizationRegistry) -> type[Any]:
    @registry.register("violin", "distribution per model")
    class Violin:
        pass

    return Violin


def test_a_new_registry_has_no_names(registry: VisualizationRegistry) -> None:
    assert registry.names == []


def test_register_returns_the_decorated_class(violin: type[Any]) -> None:
    assert violin.__name__ == "Violin"


def test_register_stamps_the_registry_name_onto_the_class(violin: type[Any]) -> None:
    assert violin.registry_name == "violin"


def test_registered_name_is_listed(registry: VisualizationRegistry, violin: type[Any]) -> None:
    assert registry.names == ["violin"]


def test_names_are_sorted(registry: VisualizationRegistry) -> None:
    registry.register("zebra")(type("Zebra", (), {}))
    registry.register("aardvark")(type("Aardvark", (), {}))

    assert registry.names == ["aardvark", "zebra"]


def test_list_names_matches_the_names_property(registry: VisualizationRegistry, violin: type[Any]) -> None:
    assert registry.list_names() == registry.names


def test_duplicate_registration_is_rejected(registry: VisualizationRegistry, violin: type[Any]) -> None:
    with pytest.raises(ValueError, match="Visualization 'violin' already registered"):

        @registry.register("violin")
        class SecondViolin:
            pass


def test_override_replaces_an_existing_name(registry: VisualizationRegistry, violin: type[Any]) -> None:
    @registry.register("violin", "replacement", override=True)
    class ReplacementViolin:
        pass

    assert registry.get("violin") is ReplacementViolin
    assert registry.describe("violin") == "replacement"


def test_the_module_facade_forwards_override(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, Any] = {}

    def fake_register(name, description, *, result_type, requirements, override):
        recorded.update(
            name=name,
            description=description,
            result_type=result_type,
            requirements=requirements,
            override=override,
        )
        return lambda cls: cls

    monkeypatch.setattr(visualization_registry, "register", fake_register)

    visualization_facade.register("x", "d", override=True)

    assert recorded["override"] is True


def test_get_returns_the_registered_class(registry: VisualizationRegistry, violin: type[Any]) -> None:
    assert registry.get("violin") is violin


def test_get_rejects_an_unknown_name(registry: VisualizationRegistry, violin: type[Any]) -> None:
    with pytest.raises(ValueError, match=r"Unknown visualization 'heatmap'\. Registered: \['violin'\]"):
        registry.get("heatmap")


def test_describe_returns_the_registered_description(registry: VisualizationRegistry, violin: type[Any]) -> None:
    assert registry.describe("violin") == "distribution per model"


def test_describe_of_an_unknown_name_is_empty(registry: VisualizationRegistry) -> None:
    assert registry.describe("heatmap") == ""


def test_describe_defaults_to_an_empty_description(registry: VisualizationRegistry) -> None:
    registry.register("bare")(type("Bare", (), {}))

    assert registry.describe("bare") == ""


def test_applicable_delegates_to_the_experiment(registry: VisualizationRegistry) -> None:
    requirements = frozenset({PlotRequirement.MULTIPLE_FOLDS})
    registry.register("folds", requirements=requirements)(type("Folds", (), {}))
    experiment = _RecordingExperiment()

    registry.applicable(experiment)  # type: ignore[arg-type]

    assert experiment.asked == [requirements]


def test_applicable_returns_classes_the_experiment_accepts(registry: VisualizationRegistry, violin: type[Any]) -> None:
    assert registry.applicable(_RecordingExperiment()) == [violin]  # type: ignore[arg-type]


def test_applicable_drops_classes_the_experiment_rejects(registry: VisualizationRegistry, violin: type[Any]) -> None:
    assert registry.applicable(_RecordingExperiment(answer=False)) == []  # type: ignore[arg-type]


def test_applicable_filters_per_class_requirements(registry: VisualizationRegistry) -> None:
    single = registry.register("single")(type("Single", (), {}))
    registry.register("comparison", requirements=frozenset({PlotRequirement.MULTIPLE_MODELS}))(
        type("Comparison", (), {})
    )

    assert registry.applicable(_SelectiveExperiment()) == [single]  # type: ignore[arg-type]


def test_applicable_on_an_empty_registry_is_empty(registry: VisualizationRegistry) -> None:
    assert registry.applicable(_RecordingExperiment()) == []  # type: ignore[arg-type]


def test_retain_only_drops_unlisted_names(registry: VisualizationRegistry, violin: type[Any]) -> None:
    registry.register("heatmap", "per-model heatmap")(type("Heatmap", (), {}))

    registry.retain_only(frozenset({"violin"}))

    assert registry.names == ["violin"]


def test_retain_only_forgets_the_dropped_description(registry: VisualizationRegistry, violin: type[Any]) -> None:
    registry.retain_only(frozenset())

    assert registry.describe("violin") == ""


def test_retain_only_frees_the_name_for_re_registration(registry: VisualizationRegistry, violin: type[Any]) -> None:
    registry.retain_only(frozenset())

    @registry.register("violin", "replacement")
    class ReplacementViolin:
        pass

    assert registry.get("violin") is ReplacementViolin


def test_repr_renders_without_an_index(registry: VisualizationRegistry, violin: type[Any]) -> None:
    rendered = repr(registry)

    assert "violin" in rendered
    assert "distribution per model" in rendered
    assert not rendered.startswith("0")


def test_repr_of_an_empty_registry_is_just_the_header(registry: VisualizationRegistry) -> None:
    assert repr(registry) == "Empty DataFrame\nColumns: []\nIndex: []"


def test_repr_html_emits_a_table(registry: VisualizationRegistry, violin: type[Any]) -> None:
    assert "<table" in registry._repr_html_()


def test_to_dataframe_lists_the_registry_columns(registry: VisualizationRegistry, violin: type[Any]) -> None:
    frame = registry.to_dataframe()

    assert list(frame.columns) == ["Name", "Description", "Result type", "Requirements"]
    assert frame.iloc[0].tolist() == ["violin", "distribution per model", "ExperimentResult", ""]


def test_to_dataframe_renders_requirements(registry: VisualizationRegistry) -> None:
    registry.register("folds", requirements=frozenset({PlotRequirement.MULTIPLE_FOLDS}))(type("Folds", (), {}))

    assert str(PlotRequirement.MULTIPLE_FOLDS) in registry.to_dataframe().iloc[0]["Requirements"]


def test_get_metadata_reports_the_registry_fields(registry: VisualizationRegistry, violin: type[Any]) -> None:
    assert registry.get_metadata("violin") == {
        "registry": "visualizations",
        "name": "violin",
        "class_name": "Violin",
        "description": "distribution per model",
        "result_type": "ExperimentResult",
        "requirements": frozenset(),
    }


def test_get_metadata_rejects_an_unknown_name(registry: VisualizationRegistry) -> None:
    with pytest.raises(ValueError, match="Unknown visualization 'heatmap'"):
        registry.get_metadata("heatmap")


def test_get_metadata_carries_the_result_type(registry: VisualizationRegistry) -> None:
    registry.register("per_model", result_type="ModelResult")(type("PerModel", (), {}))

    assert registry.get_metadata("per_model")["result_type"] == "ModelResult"


def test_list_metadata_covers_every_name(registry: VisualizationRegistry) -> None:
    registry.register("zebra")(type("Zebra", (), {}))
    registry.register("aardvark")(type("Aardvark", (), {}))

    assert [row["name"] for row in registry.list_metadata()] == ["aardvark", "zebra"]


def test_the_singleton_is_populated_by_builtin_registration() -> None:
    assert visualization_registry.names


def test_the_singleton_stamps_registry_names_onto_builtin_plots() -> None:
    name = visualization_registry.names[0]

    assert visualization_registry.get(name).registry_name == name


def test_builtin_registration_is_a_no_op_once_populated() -> None:
    from drevalpy.registry._builtins import _register_builtin_visualizations

    before = visualization_registry.names

    _register_builtin_visualizations()

    assert visualization_registry.names == before


def test_module_list_delegates_to_the_singleton() -> None:
    assert visualization_facade.list() == visualization_registry.names


def test_module_get_delegates_to_the_singleton() -> None:
    name = visualization_registry.names[0]

    assert visualization_facade.get(name) is visualization_registry.get(name)


def test_module_table_returns_a_dataframe() -> None:
    frame = visualization_facade.table()

    assert list(frame.columns) == ["Name", "Description", "Result type", "Requirements"]
    assert frame["Name"].tolist() == visualization_registry.names


def test_module_metadata_delegates_to_the_singleton() -> None:
    name = visualization_registry.names[0]

    assert visualization_facade.metadata(name) == visualization_registry.get_metadata(name)


def test_module_applicable_delegates_to_the_singleton() -> None:
    experiment = _RecordingExperiment(answer=False)

    assert visualization_facade.applicable(experiment) == []  # type: ignore[arg-type]
