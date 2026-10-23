"""Tests for the :mod:`drevalpy.registry.featurizer` package surface."""

from __future__ import annotations

import pytest

from drevalpy.errors import UnknownNameError
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.featurizer import get_featurizer
from drevalpy.registry.sample_featurizer import sample_featurizer_registry


@pytest.mark.parametrize(
    ("side", "registry"),
    [("sample", sample_featurizer_registry), ("drug", drug_featurizer_registry)],
)
def test_get_featurizer_resolves_a_registered_name_on_each_side(side: str, registry) -> None:
    name = registry.list_names()[0]

    assert get_featurizer(side, name) is registry.get(name)


def test_get_featurizer_rejects_an_unknown_side() -> None:
    with pytest.raises(UnknownNameError, match="Unknown featurizer side 'cell_line'"):
        get_featurizer("cell_line", "identity")  # type: ignore[arg-type]


@pytest.mark.parametrize("side", ["sample", "drug"])
def test_get_featurizer_rejects_an_unknown_name(side: str) -> None:
    with pytest.raises(UnknownNameError, match="Unknown .*featurizer: 'nope'"):
        get_featurizer(side, "nope")  # type: ignore[arg-type]
