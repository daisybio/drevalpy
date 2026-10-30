"""Tests for the visualization registry: registration, metadata and :func:`applicable`.

Tests register into the module registry; ``tests/conftest.py`` restores it after each test.
"""

from __future__ import annotations

import pytest

from drevalpy.errors import UnknownNameError
from drevalpy.registry._registry import Snapshot
from drevalpy.registry.visualization import applicable, get, register, visualization_registry
from drevalpy.types.results import ExperimentResult, ModelResult
from drevalpy.visualization.base import Plot
from drevalpy.visualization.requirements import PlotRequirement
from tests.synthetic import make_experiment_result


def _cls(
    name: str, *, requirements: frozenset[PlotRequirement] = frozenset(), result_type: type = ExperimentResult
) -> type[Plot]:
    return type(
        name, (Plot,), {"figure": lambda self, result: None, "requirements": requirements, "result_type": result_type}
    )


@pytest.fixture
def empty() -> None:
    visualization_registry.restore(Snapshot(visualization_registry.kind, ()))


@pytest.fixture
def violin(empty: None) -> type[Plot]:
    return register("violin", "distribution per model")(_cls("Violin"))


class TestRegistration:
    def test_register_stamps_the_name_and_keeps_registration_order(self, empty: None) -> None:
        zebra = register("zebra")(_cls("Zebra"))
        register("aardvark")(_cls("Aardvark"))

        assert zebra.name == "zebra"
        assert visualization_registry.names() == ("zebra", "aardvark")
        assert get("zebra") is zebra

    def test_a_class_that_is_not_a_plot_is_rejected(self, empty: None) -> None:
        with pytest.raises(TypeError, match="'zebra' must be a subclass of"):
            register("zebra")(type("Zebra", (), {}))

        assert visualization_registry.names() == ()

    def test_get_rejects_an_unknown_name(self, violin: type[Plot]) -> None:
        with pytest.raises(UnknownNameError, match=r"Unknown visualization 'heatmap'\. Available: violin"):
            get("heatmap")
        with pytest.raises(UnknownNameError, match="Unknown visualization 'heatmap'"):
            visualization_registry.metadata("heatmap")

    def test_duplicate_registration_is_rejected_unless_replaced(self, violin: type[Plot]) -> None:
        with pytest.raises(ValueError, match="'violin' already registered"):
            register("violin")(_cls("SecondViolin"))

        replacement = register("violin", "replacement", replace=True)(_cls("ReplacementViolin"))

        assert get("violin") is replacement
        assert visualization_registry.metadata("violin")["description"] == "replacement"

    def test_applicable_returns_the_classes_whose_requirements_the_experiment_meets(self, violin: type[Plot]) -> None:
        folds = register("folds")(_cls("Folds", requirements=frozenset({PlotRequirement.MULTIPLE_FOLDS})))
        comparison = register("comparison")(
            _cls("Comparison", requirements=frozenset({PlotRequirement.MULTIPLE_MODELS}))
        )

        assert applicable(make_experiment_result(n_models=1, n_folds=1)) == [violin]
        assert applicable(make_experiment_result(n_models=1, n_folds=2)) == [violin, folds]
        assert applicable(make_experiment_result(n_models=2, n_folds=1)) == [violin, comparison]


class TestListing:
    def test_dataframe_and_repr(self, violin: type[Plot]) -> None:
        register("folds")(_cls("Folds", requirements=frozenset({PlotRequirement.MULTIPLE_FOLDS})))

        frame = visualization_registry.to_dataframe()
        rendered = repr(visualization_registry)

        assert list(frame.columns) == ["Name", "Description", "Result type", "Requirements"]
        assert frame.iloc[0].tolist()[:3] == ["violin", "distribution per model", "ExperimentResult"]
        assert str(PlotRequirement.MULTIPLE_FOLDS) in frame.iloc[1]["Requirements"]
        assert "distribution per model" in rendered

    def test_metadata_holds_description_and_the_class_result_type_and_requirements(self, violin: type[Plot]) -> None:
        register("per_model")(_cls("PerModel", result_type=ModelResult))

        assert dict(visualization_registry.metadata("violin")) == {
            "description": "distribution per model",
            "result_type": "ExperimentResult",
            "requirements": frozenset(),
        }
        assert visualization_registry.metadata("per_model")["result_type"] == "ModelResult"


class TestBuiltins:
    def test_the_builtins_are_registered_and_stamped(self) -> None:
        names = visualization_registry.names()

        assert len(names) == 6
        assert all(get(name).name == name for name in names)

    def test_applicable_without_requirements_returns_the_unconditional_builtins(self) -> None:
        unconditional = [cls for _, cls in visualization_registry.items() if not cls.requirements]

        assert applicable(make_experiment_result(n_models=1, n_folds=1)) == unconditional
