"""Tests for the per-side featurizer binding decorator."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._side_binding import (
    derived_class_name,
    known_sides,
    register_for_sides,
)
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.featurizers.sample.base import SampleFeaturizer
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.drug_featurizer import metadata as drug_metadata
from drevalpy.registry.sample_featurizer import metadata as sample_metadata
from drevalpy.registry.sample_featurizer import sample_featurizer_registry
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _isolated_registries() -> Iterator[None]:
    yield from isolated_component_registries()


class _Probe(Featurizer):
    entity_id_only = True

    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        return self

    def _transform_features(self, source, entity_ids: np.ndarray) -> FeaturizerResult:
        return FeaturizerResult.from_blocks({}, len(entity_ids))

    @property
    def output_dim(self) -> int:
        return 1


def _bind(name: str, **kwargs) -> type:
    probe = type(_Probe.__name__, (_Probe,), {"__module__": __name__, "__doc__": _Probe.__doc__})
    return register_for_sides(name, contract=FeatureFormat.NUMERIC_MATRIX, **kwargs)(probe)


@pytest.mark.parametrize(
    ("implementation_name", "side", "expected"),
    [
        ("SharedIdentityFeaturizer", "sample", "SampleIdentityFeaturizer"),
        ("SharedIdentityFeaturizer", "drug", "DrugIdentityFeaturizer"),
        ("PlainFeaturizer", "drug", "DrugPlainFeaturizer"),
    ],
)
def test_derived_class_name_inserts_the_side_prefix(implementation_name: str, side: str, expected: str) -> None:
    assert derived_class_name(implementation_name, side) == expected


def test_known_sides_lists_both_entity_sides_and_derived_names_reject_others() -> None:
    assert known_sides() == ("drug", "sample")
    with pytest.raises(ValueError, match="unknown featurizer side"):
        derived_class_name("SharedProbeFeaturizer", "tissue")


def test_register_for_sides_binds_a_class_per_side() -> None:
    implementation = _bind("probeBoth", description="Probe.")
    sample_cls = sample_featurizer_registry.get("probeBoth")
    drug_cls = drug_featurizer_registry.get("probeBoth")

    assert sample_cls is not drug_cls
    assert (sample_cls.side, drug_cls.side) == ("sample", "drug")
    assert issubclass(sample_cls, SampleFeaturizer)
    assert issubclass(drug_cls, DrugFeaturizer)
    assert implementation.side == ""
    assert not hasattr(implementation, "registry_name")
    module = sys.modules[implementation.__module__]
    for side in known_sides():
        assert hasattr(module, derived_class_name(implementation.__name__, side))


def test_register_for_sides_accepts_per_side_descriptions_and_a_single_side() -> None:
    _bind("probeDescribed", description={"sample": "For samples.", "drug": "For drugs."})
    _bind("probeOneSide", description="Probe.", sides=("drug",))

    assert sample_metadata("probeDescribed")["description"] == "For samples."
    assert drug_metadata("probeDescribed")["description"] == "For drugs."
    assert "probeOneSide" in drug_featurizer_registry.list_names()
    assert "probeOneSide" not in sample_featurizer_registry.list_names()


def test_register_for_sides_rejects_an_unknown_side() -> None:
    with pytest.raises(ValueError, match="unknown featurizer side"):
        _bind("probeBadSide", description="Probe.", sides=("tissue",))
