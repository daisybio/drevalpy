"""Tests for the featurizer registry: the per-side registries and packages, and view-declaration validation."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import numpy as np
import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.sample.base import SampleFeaturizer
from drevalpy.errors import UnknownNameError
from drevalpy.registry import drug_featurizer, sample_featurizer
from drevalpy.registry.featurizer import (
    DrugFeaturizerRegistry,
    FeaturizerRegistry,
    SampleFeaturizerRegistry,
    drug_featurizer_registry,
    get_featurizer,
    sample_featurizer_registry,
)
from drevalpy.registry.sample_featurizer import get as get_sample_featurizer
from drevalpy.registry.sample_featurizer import register as register_sample_featurizer
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from tests.registry._helpers import isolated_component_registries

SIDES = pytest.mark.parametrize(
    ("side", "registry_cls", "singleton"),
    [
        ("sample", SampleFeaturizerRegistry, sample_featurizer_registry),
        ("drug", DrugFeaturizerRegistry, drug_featurizer_registry),
    ],
)


@SIDES
def test_side_registry_declares_its_side_and_identity(side, registry_cls, singleton) -> None:
    registry = registry_cls()

    assert issubclass(registry_cls, FeaturizerRegistry)
    assert isinstance(singleton, registry_cls)
    assert registry._side == side
    assert (registry._registry_id, registry._label, registry._display_name) == (
        f"{side}_featurizer",
        f"{side.capitalize()} featurizer",
        f"{side}_featurizers",
    )


@SIDES
def test_registration_stamps_the_side_and_keeps_an_explicit_storage_key(side, registry_cls, singleton) -> None:
    registry = registry_cls()

    @registry.register("sided", description="sided", output_format=FeatureFormat.NUMERIC_MATRIX)
    class Sided:
        storage_key = "shared_bucket"

    assert Sided.side == side
    assert Sided.storage_key == "shared_bucket"


@SIDES
def test_metadata_reports_the_side_and_listing_filters_by_tag(side, registry_cls, singleton) -> None:
    registry = registry_cls()

    @registry.register("core", description="core", output_format=FeatureFormat.NUMERIC_MATRIX)
    class Core:
        pass

    @registry.register("baseline", description="baseline", tags=("baseline",), output_format=FeatureFormat.GRAPH)
    class Baseline:
        pass

    assert registry.get_metadata("core")["registry"] == f"{side}_featurizers"
    assert {row["name"] for row in registry.list_metadata()} == {"core", "baseline"}
    tagged = registry.list_metadata(tag="baseline")
    assert [row["name"] for row in tagged] == ["baseline"]
    assert tagged[0]["tags"] == frozenset({"baseline"})


def test_isolated_registry_registers_with_format_and_restores_after_clear() -> None:
    registry = FeaturizerRegistry("test", "Test featurizer", "test_featurizers")

    @registry.register("localFeat", description="local", output_format=FeatureFormat.GRAPH)
    class LocalFeat:
        pass

    assert registry.get("localFeat") is LocalFeat
    assert vars(LocalFeat)["output_format"] == FeatureFormat.GRAPH
    assert vars(LocalFeat)["registry_name"] == "localFeat"
    assert not hasattr(LocalFeat, "side")

    registry.clear()
    assert registry.list_names() == []

    registry.register_existing("localFeat", LocalFeat)
    assert registry.get("localFeat") is LocalFeat


@pytest.mark.parametrize("side", ["sample", "drug"])
def test_get_featurizer_resolves_a_registered_name_and_rejects_an_unknown_one(side: str) -> None:
    registry = {"sample": sample_featurizer_registry, "drug": drug_featurizer_registry}[side]
    registered = registry.register("lookupDummy", description="lookup", output_format=FeatureFormat.NUMERIC_MATRIX)(
        type("LookupDummy", (), {})
    )

    assert get_featurizer(side, "lookupDummy") is registered  # type: ignore[arg-type]
    with pytest.raises(UnknownNameError, match="Unknown .*featurizer: 'nope'"):
        get_featurizer(side, "nope")  # type: ignore[arg-type]


def test_get_featurizer_rejects_an_unknown_side() -> None:
    with pytest.raises(UnknownNameError, match=r"Unknown featurizer side 'protein'.*\['sample', 'drug'\]"):
        get_featurizer("protein", "identity")  # type: ignore[arg-type]


SIDE_PACKAGES = pytest.mark.parametrize("package", [sample_featurizer, drug_featurizer], ids=["sample", "drug"])


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


def test_the_side_packages_expose_the_registries_of_the_featurizer_package() -> None:
    assert sample_featurizer.sample_featurizer_registry is sample_featurizer_registry
    assert sample_featurizer.SampleFeaturizerRegistry is SampleFeaturizerRegistry
    assert drug_featurizer.drug_featurizer_registry is drug_featurizer_registry
    assert drug_featurizer.DrugFeaturizerRegistry is DrugFeaturizerRegistry


@SIDE_PACKAGES
def test_all_names_the_registry_and_the_five_helpers(package: ModuleType) -> None:
    helpers = {"get", "list", "metadata", "register", "table"}

    assert helpers <= set(package.__all__)
    assert all(callable(getattr(package, name)) for name in helpers)


@SIDE_PACKAGES
def test_register_and_lookup(package: ModuleType) -> None:
    @package.register("dummy", description="test", output_format=FeatureFormat.NUMERIC_MATRIX)
    class Dummy:
        pass

    assert package.get("dummy") is Dummy
    assert vars(Dummy)["output_format"] == FeatureFormat.NUMERIC_MATRIX
    assert "dummy" in package.list()
    assert "dummy" in package.table()["Name"].tolist()
    meta = package.metadata("dummy")
    assert (meta["output_format"], meta["description"], meta["tags"]) == ("numeric_matrix", "test", frozenset())


@SIDE_PACKAGES
def test_duplicate_and_unknown_names_fail(package: ModuleType) -> None:
    @package.register("dup", description="first", output_format=FeatureFormat.NUMERIC_MATRIX)
    class First:
        pass

    with pytest.raises(ValueError, match="already registered"):
        package.register("dup", description="second", output_format=FeatureFormat.NUMERIC_MATRIX)(
            type("Second", (), {})
        )
    with pytest.raises(UnknownNameError, match=r"Unknown (Sample|Drug) featurizer: 'missing'"):
        package.get("missing")


@SIDE_PACKAGES
def test_the_decorator_format_overrides_the_class_body(package: ModuleType) -> None:
    @package.register("overridden", description="decorator wins", output_format=FeatureFormat.NUMERIC_MATRIX)
    class Overridden:
        output_format = FeatureFormat.GRAPH

    assert vars(Overridden)["output_format"] == FeatureFormat.NUMERIC_MATRIX


@SIDE_PACKAGES
def test_a_class_body_format_is_a_valid_declaration(package: ModuleType) -> None:
    @package.register("bodyFormat", description="declares on the class body")
    class BodyFormat:
        output_format = FeatureFormat.GRAPH

    assert package.metadata("bodyFormat")["output_format"] == "graph"


@SIDE_PACKAGES
@pytest.mark.parametrize(
    "body",
    [{}, {"output_format": "numeric_matrix_but_a_plain_string"}],
    ids=["missing", "wrong-type"],
)
def test_a_missing_or_invalid_format_is_rejected(package: ModuleType, body: dict) -> None:
    with pytest.raises(TypeError, match="BadFormat.output_format must be a FeatureFormat"):
        package.register("badFormat", description="bad")(type("BadFormat", (), body))


class _ConcreteSampleFeaturizer(SampleFeaturizer):
    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        return self

    def _transform_features(self, source, entity_ids):
        values = np.zeros((len(entity_ids), 1), dtype=np.float32)
        return FeaturizerResult(values, {"probe": numeric_feature_block(values)})

    @property
    def output_dim(self) -> int:
        return 1


def _register(name: str, cls: type) -> type:
    return register_sample_featurizer(name, description=name, output_format=FeatureFormat.NUMERIC_MATRIX)(cls)


def test_featurizer_registration_requires_declared_input_views() -> None:
    with pytest.raises(ValueError, match="does not declare its input views"):
        _register("undeclaredViews", type("UndeclaredViews", (_ConcreteSampleFeaturizer,), {}))

    _register("declaredViews", type("DeclaredViews", (_ConcreteSampleFeaturizer,), {"input_views": ("methylation",)}))

    assert get_sample_featurizer("declaredViews").resolve_input_views() == ("methylation",)


def test_a_featurizer_with_unimplemented_abstract_methods_is_rejected() -> None:
    abstract = type(
        "AbstractFeaturizer",
        (SampleFeaturizer,),
        {"input_views": ("methylation",), "output_dim": property(lambda self: 0)},
    )

    with pytest.raises(ValueError, match=r"does not implement _fit, _transform_features"):
        _register("abstractFeaturizer", abstract)
