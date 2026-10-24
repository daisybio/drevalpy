"""Tests for the drug featurizer base class (``drug/base``)."""

from __future__ import annotations

import pytest

from drevalpy.components.featurizers._dense_view import DenseViewFeaturizer
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.featurizers.drug.base import DenseViewDrugFeaturizer, DrugFeaturizer
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.registry.drug_featurizer import list as list_drug_featurizers


def test_drug_featurizer_is_abstract_and_extends_the_shared_base() -> None:
    assert issubclass(DrugFeaturizer, Featurizer)
    with pytest.raises(TypeError, match="abstract"):
        DrugFeaturizer()


def test_every_registered_drug_featurizer_derives_from_the_base() -> None:
    names = list_drug_featurizers()
    assert names
    assert all(issubclass(get_drug_featurizer(name), DrugFeaturizer) for name in names)


def test_dense_view_binding_resolves_the_side_base_before_the_shared_featurizer() -> None:
    mro = DenseViewDrugFeaturizer.__mro__
    assert issubclass(DenseViewDrugFeaturizer, DenseViewFeaturizer)
    assert mro.index(DrugFeaturizer) < mro.index(Featurizer)
