"""Tests for the registered single-view drug featurizer.

Mirrors :mod:`drevalpy.components.featurizers.drug.view`, which is now a thin
registered binding over ``DenseViewFeaturizer``; the shared fit/transform/block
behaviour is covered once in ``tests/components/featurizers/test_dense_view.py``.
What is specific to this module is its registration, its default view, and the
fact that the three on-the-fly drug featurizers no longer inherit from it.
"""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers._dense_view import DenseViewFeaturizer
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from tests.conftest import MockFeatureSource


def _drug_source() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "d1": {"morgan_fingerprint": np.array([1.0, 0.0])},
            "d2": {"morgan_fingerprint": np.array([0.0, 1.0])},
        },
        meta_info={"morgan_fingerprint": ["fp1", "fp2"]},
    )


def test_view_drug_featurizer_is_registered_under_view() -> None:
    assert get_drug_featurizer("view") is ViewDrugFeaturizer
    assert ViewDrugFeaturizer.side == "drug"


def test_view_drug_featurizer_defaults_to_the_morgan_fingerprint_view() -> None:
    assert ViewDrugFeaturizer.input_views == ("morgan_fingerprint",)
    assert ViewDrugFeaturizer()._view == "morgan_fingerprint"


def test_view_drug_featurizer_reuses_the_shared_dense_base() -> None:
    assert issubclass(ViewDrugFeaturizer, DenseViewFeaturizer)
    assert "_transform" not in vars(ViewDrugFeaturizer)


def test_view_drug_featurizer_passes_the_view_through() -> None:
    source = _drug_source()
    ids = np.array(["d1", "d2"], dtype=str)
    featurizer = ViewDrugFeaturizer().fit(source, entity_ids=ids)

    blocks = featurizer.transform_blocks(source, ids)

    assert featurizer.output_dim == 2
    assert set(blocks) == {"morgan_fingerprint"}
    assert blocks["morgan_fingerprint"].feature_names == ("fp1", "fp2")
    np.testing.assert_allclose(featurizer.transform(source, ids), [[1.0, 0.0], [0.0, 1.0]])
