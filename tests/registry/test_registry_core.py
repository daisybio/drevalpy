"""Tests for the shared ``Registry`` ABC and the built-in component registration."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.errors import UnknownNameError
from drevalpy.registry._base import Registry, TableReprMixin
from drevalpy.registry._builtins import _registry_for
from drevalpy.registry.dataset import dataset_registry
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.registry.drug_featurizer import list as list_drug_featurizers
from drevalpy.registry.drug_featurizer import metadata as get_drug_featurizer_metadata
from drevalpy.registry.featurizer import drug_featurizer_registry, sample_featurizer_registry
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.registry.predictor._registry import predictor_registry
from drevalpy.registry.sample_featurizer import get as get_sample_featurizer
from drevalpy.registry.sample_featurizer import list as list_sample_featurizers
from drevalpy.registry.splitter import splitter_registry
from drevalpy.registry.visualization import visualization_registry
from tests._trusted_subprocess import run_trusted_python
from tests.registry._helpers import restore_component_registries


class _ExampleRegistry(Registry):
    def __init__(self) -> None:
        super().__init__("example", "Example component", "examples")

    def add(self, name: str, cls: type[Any]) -> None:
        self._store[name] = cls

    def _component_metadata(self, name: str, cls: type[Any]) -> dict[str, Any]:
        return {
            "registry": self._display_name,
            "name": name,
            "description": getattr(cls, "description", ""),
            "tags": getattr(cls, "tags", frozenset()),
        }


class _Alpha:
    description = "first component"
    tags = frozenset({"baseline"})


class _Beta:
    description = "second component"
    tags = frozenset({"omics", "baseline"})


@pytest.fixture
def registry() -> _ExampleRegistry:
    populated = _ExampleRegistry()
    populated.add("alpha", _Alpha)
    populated.add("beta", _Beta)
    return populated


def test_registry_is_abstract() -> None:
    with pytest.raises(TypeError, match="_component_metadata"):
        Registry("example", "Example", "examples")  # type: ignore[abstract]


def test_get_and_list_names(registry: _ExampleRegistry) -> None:
    assert _ExampleRegistry().list_names() == []
    assert registry.list_names() == ["alpha", "beta"]
    assert registry.get("alpha") is _Alpha
    with pytest.raises(
        UnknownNameError, match=r"Unknown Example component: 'missing'\. Available: \['alpha', 'beta'\]"
    ):
        registry.get("missing")


def test_metadata_listing_and_tag_filter(registry: _ExampleRegistry) -> None:
    assert registry.get_metadata("alpha") == {
        "registry": "examples",
        "name": "alpha",
        "description": "first component",
        "tags": frozenset({"baseline"}),
    }
    assert [row["name"] for row in registry.list_metadata()] == ["alpha", "beta"]
    assert [row["name"] for row in registry.list_metadata(tag="  omics  ")] == ["beta"]


def test_clear_and_retain_only(registry: _ExampleRegistry) -> None:
    registry.retain_only(frozenset({"beta", "never-registered"}))
    assert registry.list_names() == ["beta"]

    registry.clear()
    assert registry.list_names() == []


def test_dataframe_repr_and_html(registry: _ExampleRegistry) -> None:
    frame = registry.to_dataframe()

    assert list(frame.columns) == ["Name", "Description", "Tags"]
    pd.testing.assert_series_equal(frame["Tags"], pd.Series(["baseline", "baseline, omics"], name="Tags"))
    assert _ExampleRegistry().to_dataframe().empty
    assert "alpha" in repr(registry)
    assert not repr(registry).startswith("0")
    assert "<table" in registry._repr_html_()


@pytest.mark.parametrize(
    "registry",
    [dataset_registry, splitter_registry, visualization_registry],
    ids=["dataset", "splitter", "visualization"],
)
def test_registries_outside_the_component_hierarchy_share_the_table_repr(registry: TableReprMixin) -> None:
    assert isinstance(registry, TableReprMixin)
    assert "<table" in registry._repr_html_()


@pytest.fixture(autouse=True)
def _register_components() -> Iterator[None]:
    restore_component_registries()
    yield
    restore_component_registries()


@pytest.mark.parametrize(
    ("names", "get"),
    [(list_sample_featurizers, get_sample_featurizer), (list_drug_featurizers, get_drug_featurizer)],
    ids=["sample", "drug"],
)
def test_builtin_featurizers_declare_an_output_format(names, get) -> None:
    for name in names():
        cls = get(name)
        assert "output_format" in cls.__dict__, name
        assert isinstance(cls.output_format, FeatureFormat)


def test_bpe_pharmaformer_has_literature_reference() -> None:
    meta = get_drug_featurizer_metadata("bpePharmaformer")

    assert meta["repo_url"] == "https://github.com/zhouyuru1205/PharmaFormer"
    assert meta["citation_doi"] == "10.1038/s41698-025-01082-6"
    assert meta["citation"].startswith("https://doi.org/10.1038/")
    assert meta["deviations"]


@pytest.mark.slow
def test_fresh_process_discovery_returns_all_builtins() -> None:
    script = """
from drevalpy.registry.sample_featurizer import sample_featurizer_registry
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.predictor import predictor_registry
assert len(sample_featurizer_registry.list_metadata()) == 18
assert len(drug_featurizer_registry.list_metadata()) == 12
assert len(predictor_registry.list_metadata()) == 30
print("ok")
"""
    completed = run_trusted_python(script)

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


@pytest.mark.parametrize(
    "name", ["precily", "srmf", "molir", "superfeltr", "pharmaFormer", "dipk", "sparsego", "naiveMean", "naiveDrugMean"]
)
def test_split_module_predictors_register_under_their_name(name: str) -> None:
    assert get_predictor(name).registry_name == name


class TestRegistryDispatchOnRescan:
    """``_registry_for`` recovers a class's registry from what registration stamped on it."""

    def test_registered_components_go_back_to_their_own_registry(self) -> None:
        assert _registry_for(get_sample_featurizer(list_sample_featurizers()[0])) is sample_featurizer_registry
        assert _registry_for(get_drug_featurizer(list_drug_featurizers()[0])) is drug_featurizer_registry
        assert _registry_for(get_predictor("naiveMean")) is predictor_registry

    def test_the_side_takes_precedence_over_a_predictor_format(self) -> None:
        class Ambiguous:
            side = "drug"
            sample_input_format = object()

        assert _registry_for(Ambiguous) is drug_featurizer_registry

    def test_a_sideless_featurizer_falls_back_to_the_sample_registry(self) -> None:
        class Sideless:
            output_format = object()

        assert _registry_for(Sideless) is sample_featurizer_registry

    def test_a_plain_class_belongs_to_no_registry(self) -> None:
        assert _registry_for(type("NotAComponent", (), {})) is None
