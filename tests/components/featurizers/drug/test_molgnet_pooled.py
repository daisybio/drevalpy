"""Tests for the pooled MolGNet drug featurizer; runs offline with a mock source and stubbed embedding."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.drug import molgnet
from drevalpy.components.featurizers.drug.molgnet_pooled import (
    MolGNetPooledFeaturizer,
    molgnet_feature_names,
    pool_molgnet,
)
from drevalpy.errors import InvalidInputError
from tests.conftest import MockFeatureSource

_D1 = np.arange(6, dtype=np.float32).reshape(2, 3)
_D2 = np.array([[1.0, -2.0, 4.0], [3.0, 0.0, -1.0], [-5.0, 8.0, 2.0]], dtype=np.float32)
_IDS = np.array(["d1", "d2"])


def _source() -> MockFeatureSource:
    return MockFeatureSource({"d1": {"molgnet_features": _D1}, "d2": {"molgnet_features": _D2}})


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [
        ("mean_max", np.concatenate([_D2.mean(axis=0), _D2.max(axis=0)])),
        ("mean", _D2.mean(axis=0)),
        ("max", _D2.max(axis=0)),
    ],
)
def test_pool_molgnet_reductions(pooling: str, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(pool_molgnet(_D2, pooling), expected)


def test_unknown_pooling_is_rejected_by_the_function_and_the_constructor() -> None:
    with pytest.raises(InvalidInputError, match="Unknown MolGNet pooling 'sum'"):
        pool_molgnet(_D2, "sum")
    with pytest.raises(InvalidInputError, match="mean_max, mean, max"):
        MolGNetPooledFeaturizer(pooling="median")


def test_molgnet_feature_names_follow_the_pooling() -> None:
    assert molgnet_feature_names(4, "mean_max") == (
        "molgnet_mean_0",
        "molgnet_mean_1",
        "molgnet_max_0",
        "molgnet_max_1",
    )
    assert molgnet_feature_names(2, "max") == ("molgnet_max_0", "molgnet_max_1")


@pytest.mark.parametrize(("pooling", "width"), [("mean_max", 6), ("mean", 3), ("max", 3)])
def test_transform_pools_every_drug_and_reports_the_width(pooling: str, width: int) -> None:
    source = _source()
    featurizer = MolGNetPooledFeaturizer(pooling=pooling).fit(source, entity_ids=_IDS)

    matrix = featurizer.transform(source, _IDS)

    assert featurizer.output_dim == width
    assert matrix.shape == (2, width)
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[1], pool_molgnet(_D2, pooling))
    assert featurizer.transform(source, np.array([], dtype=str)).shape == (0, width)


def test_transform_blocks_emits_one_named_numeric_block() -> None:
    source = _source()
    featurizer = MolGNetPooledFeaturizer().fit(source, entity_ids=_IDS)

    blocks = featurizer.transform_blocks(source, _IDS)

    assert set(blocks) == {"molgnet_pooled"}
    assert blocks["molgnet_pooled"].format is FeatureFormat.NUMERIC_MATRIX
    assert blocks["molgnet_pooled"].feature_names == molgnet_feature_names(6, "mean_max")


def test_equal_atom_counts_still_pool_per_drug_and_unfitted_drugs_are_read() -> None:
    source = MockFeatureSource(
        {"a": {"molgnet_features": _D1}, "b": {"molgnet_features": _D1 + 10}, "c": {"molgnet_features": _D2}}
    )
    featurizer = MolGNetPooledFeaturizer(pooling="mean").fit(source, entity_ids=np.array(["a", "b"]))

    matrix = featurizer.transform(source, np.array(["a", "b", "c"]))

    np.testing.assert_array_equal(matrix[:2], np.stack([_D1.mean(axis=0), (_D1 + 10).mean(axis=0)]))
    np.testing.assert_array_equal(matrix[2], _D2.mean(axis=0))


def test_missing_view_is_computed_from_the_smiles(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def compute(self: MolGNetPooledFeaturizer, source, drug_id: str) -> np.ndarray:
        calls.append(drug_id)
        return _D1

    monkeypatch.setattr(MolGNetPooledFeaturizer, "_compute_single_embedding", compute)
    source = MockFeatureSource({"d1": {}, "d2": {"molgnet_features": _D2}})
    featurizer = MolGNetPooledFeaturizer(pooling="mean").fit(source, entity_ids=_IDS)

    assert calls == ["d1"]
    np.testing.assert_array_equal(featurizer.transform(source, np.array(["d1"]))[0], _D1.mean(axis=0))


def test_unavailable_drug_raises_a_key_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(molgnet.MolGNetDrugFeaturizer, "_compute_single_embedding", lambda *_: None)
    unavailable = MockFeatureSource({"d1": {"molgnet_features": _D1}, "d2": {}})
    fitted = MolGNetPooledFeaturizer().fit(unavailable, entity_ids=np.array(["d1"]))
    with pytest.raises(KeyError, match="d2"):
        fitted.transform(unavailable, np.array(["d2"]))


def test_hyperparameter_space_lists_the_poolings() -> None:
    assert MolGNetPooledFeaturizer.get_hyperparameter_space()["pooling"]["choices"] == ["mean_max", "mean", "max"]
    assert MolGNetPooledFeaturizer.get_default_hyperparameters() == {"pooling": "mean_max"}
