"""Tests for the cell-line featurizer base class.

Mirrors :mod:`drevalpy.components.featurizers.cell_line.base`, a three-statement
subclass of ``Featurizer``: the only behaviour it carries is its position in the
MRO, which registration and the config layer both key off.
"""

from __future__ import annotations

import pytest

from drevalpy.components.featurizers._dense_view import DenseViewFeaturizer
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer, DenseViewCellLineFeaturizer
from drevalpy.registry.cell_line_featurizer import get as get_cell_line_featurizer
from drevalpy.registry.cell_line_featurizer import list as list_cell_line_featurizers


def test_cell_line_featurizer_extends_the_shared_base() -> None:
    assert issubclass(CellLineFeaturizer, Featurizer)


def test_cell_line_featurizer_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        CellLineFeaturizer()


def test_cell_line_featurizer_adds_no_state_of_its_own() -> None:
    assert set(CellLineFeaturizer.__dict__) - set(Featurizer.__dict__) <= {
        "__module__",
        "__doc__",
        "__abstractmethods__",
        "_abc_impl",
        "__firstlineno__",
        "__static_attributes__",
    }


def test_every_registered_cell_line_featurizer_derives_from_the_base() -> None:
    names = list_cell_line_featurizers()

    assert names
    for name in names:
        assert issubclass(get_cell_line_featurizer(name), CellLineFeaturizer), name


def test_dense_view_binding_sits_on_both_the_shared_base_and_the_side_base() -> None:
    assert issubclass(DenseViewCellLineFeaturizer, DenseViewFeaturizer)
    assert issubclass(DenseViewCellLineFeaturizer, CellLineFeaturizer)


def test_dense_view_binding_resolves_the_side_base_before_the_shared_featurizer() -> None:
    """The MRO order is what lets the cell-line base override shared behaviour."""
    mro = DenseViewCellLineFeaturizer.__mro__

    assert mro.index(CellLineFeaturizer) < mro.index(Featurizer)
