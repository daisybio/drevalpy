"""Tests for the featurizer registries: registration on one side, declaration validation and the lookup."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.errors import UnknownNameError
from drevalpy.registry._component import featurizer_component_metadata
from drevalpy.registry.featurizer import featurizer_registry, get_featurizer, names_for_side, register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import Side
from tests.registry._helpers import empty_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> None:
    empty_component_registries()


class _Probe(Featurizer):
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


def test_each_side_has_its_own_registry() -> None:
    assert featurizer_registry(Side.SAMPLE).kind == "sample featurizer"
    assert featurizer_registry("drug").kind == "drug featurizer"


def test_registration_sets_the_name_and_side_and_stores_description_and_tags_only_in_the_registry() -> None:
    cls = register("plainKey", side=Side.SAMPLE, description="plain")(_subclass("PlainKey"))

    assert featurizer_registry("sample").get("plainKey") is cls
    assert (cls.name, cls.side) == ("plainKey", Side.SAMPLE)
    assert vars(cls)["reference"] is None
    assert not {"description", "tags", "registry_name"} & set(vars(cls))


def test_registry_metadata_is_description_and_tags() -> None:
    cls = register("baseline", side=Side.SAMPLE, description="baseline", tags=("baseline",))(_subclass("Baseline"))

    assert dict(featurizer_registry("sample").metadata("baseline")) == {
        "description": "baseline",
        "tags": frozenset({"baseline"}),
    }
    catalog = featurizer_component_metadata("featurizer", "baseline", cls)
    assert (catalog["registry"], catalog["output_format"], catalog["side"]) == (
        "featurizer",
        "numeric_matrix",
        "sample",
    )


def test_duplicate_names_are_rejected_within_a_side() -> None:
    register("dup", side=Side.SAMPLE, description="first")(_subclass("First"))

    with pytest.raises(ValueError, match="already registered"):
        register("dup", side=Side.SAMPLE, description="second")(_subclass("Second"))


def test_one_name_can_be_registered_on_both_sides() -> None:
    sample = register("shared", side=Side.SAMPLE, description="s")(_subclass("SampleShared"))
    drug = register("shared", side=Side.DRUG, description="d")(_subclass("DrugShared"))

    assert get_featurizer("shared", "sample") is sample
    assert get_featurizer("shared", "drug") is drug
    assert (sample.side, drug.side) == (Side.SAMPLE, Side.DRUG)


def test_get_featurizer_looks_up_the_side() -> None:
    cls = register("sampleOnly", side=Side.SAMPLE, description="sample only")(_subclass("SampleOnly"))

    assert get_featurizer("sampleOnly", Side.SAMPLE) is cls
    with pytest.raises(
        UnknownNameError, match="'sampleOnly' is a sample featurizer and cannot be used on the drug side"
    ):
        get_featurizer("sampleOnly", "drug")


def test_an_unknown_side_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown featurizer side 'protein'"):
        get_featurizer("known", "protein")
    with pytest.raises(ValueError, match="Unknown featurizer side 'protein'"):
        register("bad", side="protein", description="bad")  # type: ignore[arg-type]


def test_names_for_side_lists_that_sides_registry() -> None:
    register("onlySample", side=Side.SAMPLE, description="s")(_subclass("OnlySample"))
    register("onlyDrug", side=Side.DRUG, description="d")(_subclass("OnlyDrug"))

    assert names_for_side("sample") == ("onlySample",)
    assert names_for_side("drug") == ("onlyDrug",)


def test_a_non_featurizer_class_is_rejected() -> None:
    with pytest.raises(TypeError, match="must be a subclass of"):
        register("plain", side=Side.SAMPLE, description="plain")(type("Plain", (), {}))


@pytest.mark.parametrize("attr", ["input_views", "source_views"])
@pytest.mark.parametrize("views", [["methylation"], ("",), (1,)], ids=["list", "empty-name", "not-a-string"])
def test_malformed_view_declarations_are_rejected(attr: str, views) -> None:
    with pytest.raises(TypeError, match=f"{attr} must be a tuple of view names"):
        register("badViews", side=Side.SAMPLE, description="bad")(_subclass("BadViews", **{attr: views}))


def test_a_featurizer_with_unimplemented_abstract_methods_is_rejected() -> None:
    abstract = type("AbstractFeaturizer", (Featurizer,), {})

    with pytest.raises(ValueError, match=r"does not implement fit, transform"):
        register("abstractFeaturizer", side=Side.SAMPLE, description="abstract")(abstract)
