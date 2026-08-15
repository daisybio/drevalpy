"""Tests for the per-side featurizer binding decorator.

Mirrors :mod:`drevalpy.components.featurizers._side_binding`. The tests that register
names go through ``isolated_component_registries`` because the component registries
are process-global; ``register_builtin_components`` only adds, so a registration left
behind resurfaces much later as a duplicate-name ``ValueError``.
"""

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
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.registry.cell_line_featurizer import cell_line_featurizer_registry
from drevalpy.registry.cell_line_featurizer import metadata as cell_line_metadata
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.drug_featurizer import metadata as drug_metadata
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _isolated_registries() -> Iterator[None]:
    """Register into empty registries and restore the built-ins afterwards."""
    yield from isolated_component_registries()


class _Probe(Featurizer):
    """Minimal side-agnostic implementation used as the decorator's input."""

    entity_id_only = True

    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        """Fitting is a no-op."""
        return self

    def _transform_blocks(self, source, entity_ids: np.ndarray) -> dict:
        """No blocks; the probe is only ever inspected as a class."""
        return {}

    @property
    def output_dim(self) -> int:
        """Fixed width."""
        return 1


def _bind(name: str, **kwargs) -> type:
    """Apply the decorator to a fresh copy of the probe under *name*."""
    probe = type(_Probe.__name__, (_Probe,), {"__module__": __name__, "__doc__": _Probe.__doc__})
    return register_for_sides(name, contract=FeatureFormat.NUMERIC_MATRIX, **kwargs)(probe)


def test_known_sides_lists_both_entity_sides() -> None:
    assert known_sides() == ("cell_line", "drug")


@pytest.mark.parametrize(
    ("implementation_name", "side", "expected"),
    [
        ("SharedIdentityFeaturizer", "cell_line", "CellLineIdentityFeaturizer"),
        ("SharedIdentityFeaturizer", "drug", "DrugIdentityFeaturizer"),
        ("PlainFeaturizer", "drug", "DrugPlainFeaturizer"),
    ],
)
def test_derived_class_name_inserts_the_side_prefix(implementation_name: str, side: str, expected: str) -> None:
    assert derived_class_name(implementation_name, side) == expected


def test_derived_class_name_rejects_an_unknown_side() -> None:
    with pytest.raises(ValueError, match="unknown featurizer side"):
        derived_class_name("SharedProbeFeaturizer", "tissue")


def test_register_for_sides_registers_on_both_sides() -> None:
    _bind("probeBoth", description="Probe.")

    assert "probeBoth" in cell_line_featurizer_registry.list_names()
    assert "probeBoth" in drug_featurizer_registry.list_names()


def test_register_for_sides_gives_each_side_its_own_class_and_side_value() -> None:
    _bind("probeSides", description="Probe.")

    cell_line_cls = cell_line_featurizer_registry.get("probeSides")
    drug_cls = drug_featurizer_registry.get("probeSides")

    assert cell_line_cls is not drug_cls
    assert cell_line_cls.side == "cell_line"
    assert drug_cls.side == "drug"


def test_register_for_sides_binds_each_side_to_its_base_class() -> None:
    _bind("probeBases", description="Probe.")

    assert issubclass(cell_line_featurizer_registry.get("probeBases"), CellLineFeaturizer)
    assert issubclass(drug_featurizer_registry.get("probeBases"), DrugFeaturizer)


def test_register_for_sides_returns_the_unregistered_implementation() -> None:
    implementation = _bind("probeReturn", description="Probe.")

    assert implementation.side == ""
    assert not hasattr(implementation, "registry_name")


def test_register_for_sides_injects_the_derived_classes_into_the_module() -> None:
    """``_reregister_from_module`` walks ``vars(module)``, so the classes must land there."""
    implementation = _bind("probeInject", description="Probe.")
    module = sys.modules[implementation.__module__]

    for side in known_sides():
        assert hasattr(module, derived_class_name(implementation.__name__, side))


def test_register_for_sides_accepts_a_per_side_description() -> None:
    _bind("probeDescribed", description={"cell_line": "For cell lines.", "drug": "For drugs."})

    assert cell_line_metadata("probeDescribed")["description"] == "For cell lines."
    assert drug_metadata("probeDescribed")["description"] == "For drugs."


def test_register_for_sides_can_bind_a_single_side() -> None:
    _bind("probeOneSide", description="Probe.", sides=("drug",))

    assert "probeOneSide" in drug_featurizer_registry.list_names()
    assert "probeOneSide" not in cell_line_featurizer_registry.list_names()


def test_register_for_sides_rejects_an_unknown_side() -> None:
    with pytest.raises(ValueError, match="unknown featurizer side"):
        _bind("probeBadSide", description="Probe.", sides=("tissue",))
