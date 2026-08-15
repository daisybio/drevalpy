"""Tests for the drug featurizer base class.

Mirrors :mod:`drevalpy.components.featurizers.drug.base`, a three-statement
subclass of ``Featurizer``: the only behaviour it carries is its position in the
MRO, which registration and the config layer both key off.
"""

from __future__ import annotations

import pytest

from drevalpy.components.featurizers._dense_view import DenseViewFeaturizer
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.featurizers.drug.base import DenseViewDrugFeaturizer, DrugFeaturizer
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.registry.drug_featurizer import list as list_drug_featurizers


def test_drug_featurizer_extends_the_shared_base() -> None:
    assert issubclass(DrugFeaturizer, Featurizer)


def test_drug_featurizer_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        DrugFeaturizer()


def test_drug_featurizer_adds_no_state_of_its_own() -> None:
    assert set(DrugFeaturizer.__dict__) - set(Featurizer.__dict__) <= {
        "__module__",
        "__doc__",
        "__abstractmethods__",
        "_abc_impl",
        "__firstlineno__",
        "__static_attributes__",
    }


def test_every_registered_drug_featurizer_derives_from_the_base() -> None:
    names = list_drug_featurizers()

    assert names
    for name in names:
        assert issubclass(get_drug_featurizer(name), DrugFeaturizer), name


def test_dense_view_binding_sits_on_both_the_shared_base_and_the_side_base() -> None:
    assert issubclass(DenseViewDrugFeaturizer, DenseViewFeaturizer)
    assert issubclass(DenseViewDrugFeaturizer, DrugFeaturizer)


def test_dense_view_binding_resolves_the_side_base_before_the_shared_featurizer() -> None:
    """The MRO order is what lets the drug base override shared behaviour."""
    mro = DenseViewDrugFeaturizer.__mro__

    assert mro.index(DrugFeaturizer) < mro.index(Featurizer)
