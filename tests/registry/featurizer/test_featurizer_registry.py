"""Tests for the featurizer registry: registration, declaration validation and the side-checking lookup."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.errors import UnknownNameError
from drevalpy.registry.components import featurizer_component_metadata
from drevalpy.registry.featurizer import featurizers, get_featurizer, names_for_side, register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from tests.registry._helpers import empty_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> None:
    empty_component_registries()


class _Probe(Featurizer):
    sides = frozenset({"sample"})

    @classmethod
    def output_blocks(cls, config) -> tuple[BlockSpec, ...]:
        return (BlockSpec("probe", FeatureFormat.NUMERIC_MATRIX),)

    def fit(self, source, train_ids, *, train_pair_ids=None, es_pair_ids=None):
        return self

    def transform(self, source, ids) -> FeaturizerResult:
        values = np.zeros((len(ids), 1), dtype=np.float32)
        return FeaturizerResult(ids, {"probe": numeric_feature_block(values)}, ("probe",))


def _subclass(name: str, **attrs) -> type[Featurizer]:
    return type(name, (_Probe,), attrs)


def test_the_registry_has_one_featurizer_kind() -> None:
    assert featurizers.kind == "featurizer"


def test_registration_sets_the_name_and_defaults_the_storage_key_to_it() -> None:
    cls = register("plainKey", description="plain")(_subclass("PlainKey"))

    assert featurizers.get("plainKey") is cls
    assert cls.name == "plainKey"
    assert cls.storage_key == "plainKey"
    assert vars(cls)["registry_name"] == "plainKey"


def test_registration_keeps_an_explicit_storage_key() -> None:
    cls = register("keyed", description="keyed")(_subclass("Keyed", storage_key="shared_bucket"))

    assert cls.storage_key == "shared_bucket"


def test_registry_metadata_is_description_and_tags() -> None:
    cls = register("baseline", description="baseline", tags=("baseline",))(_subclass("Baseline"))

    assert dict(featurizers.metadata("baseline")) == {"description": "baseline", "tags": frozenset({"baseline"})}
    catalog = featurizer_component_metadata("featurizer", "baseline", cls)
    assert (catalog["registry"], catalog["output_format"], catalog["sides"]) == (
        "featurizer",
        "numeric_matrix",
        frozenset({"sample"}),
    )


def test_duplicate_names_are_rejected() -> None:
    register("dup", description="first")(_subclass("First"))

    with pytest.raises(ValueError, match="already registered"):
        register("dup", description="second")(_subclass("Second"))


def test_get_featurizer_checks_the_side() -> None:
    cls = register("sampleOnly", description="sample only")(_subclass("SampleOnly"))

    assert get_featurizer("sampleOnly") is cls
    assert get_featurizer("sampleOnly", "sample") is cls
    with pytest.raises(
        ValueError, match="Featurizer 'sampleOnly' cannot be used on the drug side; it supports only: sample"
    ):
        get_featurizer("sampleOnly", "drug")


def test_get_featurizer_rejects_an_unknown_name_and_an_unknown_side() -> None:
    register("known", description="known")(_subclass("Known"))

    with pytest.raises(UnknownNameError, match="Unknown featurizer 'nope'"):
        get_featurizer("nope", "sample")
    with pytest.raises(ValueError, match="Unknown featurizer side 'protein'"):
        get_featurizer("known", "protein")  # type: ignore[arg-type]


def test_names_for_side_filters_by_declared_sides() -> None:
    register("onlySample", description="s")(_subclass("OnlySample"))
    register("onlyDrug", description="d")(_subclass("OnlyDrug", sides=frozenset({"drug"})))
    register("both", description="b")(_subclass("Both", sides=frozenset({"sample", "drug"})))

    assert names_for_side("sample") == ("onlySample", "both")
    assert names_for_side("drug") == ("onlyDrug", "both")


def test_a_non_featurizer_class_is_rejected() -> None:
    with pytest.raises(TypeError, match="must be a subclass of"):
        register("plain", description="plain")(type("Plain", (), {}))


@pytest.mark.parametrize(
    "sides",
    [frozenset(), frozenset({"protein"}), {"sample"}],
    ids=["empty", "unknown-side", "not-a-frozenset"],
)
def test_invalid_sides_are_rejected(sides) -> None:
    with pytest.raises(ValueError, match="sides must be a non-empty frozenset"):
        register("badSides", description="bad")(_subclass("BadSides", sides=sides))


@pytest.mark.parametrize("attr", ["input_views", "source_views"])
@pytest.mark.parametrize("views", [["methylation"], ("",), (1,)], ids=["list", "empty-name", "not-a-string"])
def test_malformed_view_declarations_are_rejected(attr: str, views) -> None:
    with pytest.raises(TypeError, match=f"{attr} must be a tuple of view names"):
        register("badViews", description="bad")(_subclass("BadViews", **{attr: views}))


def test_a_featurizer_with_unimplemented_abstract_methods_is_rejected() -> None:
    abstract = type("AbstractFeaturizer", (Featurizer,), {"sides": frozenset({"sample"})})

    with pytest.raises(ValueError, match=r"does not implement fit, transform"):
        register("abstractFeaturizer", description="abstract")(abstract)
