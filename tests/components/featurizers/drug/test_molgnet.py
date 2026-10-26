"""Tests for the MolGNet drug featurizers; only the checkpoint load is network-bound."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import drevalpy.components.featurizers.drug.molgnet as molgnet_module
from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.drug import molgnet
from drevalpy.components.featurizers.drug.molgnet import (
    MolGNetDrugFeaturizer,
    _compute_molgnet_embedding,
)
from drevalpy.components.featurizers.drug.molgnet_pooled import (
    MolGNetPooledFeaturizer,
    molgnet_feature_names,
    pool_molgnet,
)
from drevalpy.errors import InvalidInputError
from drevalpy.types.data.feature_source import DrugFeatureSource
from tests._import_shims import block_imports
from tests.conftest import MockFeatureSource

_D1 = np.arange(6, dtype=np.float32).reshape(2, 3)
_D2 = np.array([[1.0, -2.0, 4.0], [3.0, 0.0, -1.0], [-5.0, 8.0, 2.0]], dtype=np.float32)
_IDS = np.array(["d1", "d2"], dtype=str)


def _ragged_source() -> MockFeatureSource:
    return MockFeatureSource(features={"d1": {"molgnet_features": _D1}, "d2": {"molgnet_features": _D2}})


class _NoStoredTensorSource(DrugFeatureSource):
    def get_entity_view(self, entity_id: str, view: str) -> None:
        return None


def _patch_embedding(monkeypatch: pytest.MonkeyPatch, rows: int = 4) -> np.ndarray:
    embedding = np.ones((rows, 3), dtype=np.float32)
    monkeypatch.setattr(molgnet_module, "_compute_molgnet_embedding", lambda smiles: embedding)
    return embedding


def test_molgnet_infers_width_and_returns_one_ragged_tensor_per_drug() -> None:
    source = _ragged_source()
    featurizer = MolGNetDrugFeaturizer()
    assert featurizer.output_dim == 0
    featurizer.fit(source, entity_ids=_IDS)

    payloads = featurizer.transform(source, _IDS)
    blocks = featurizer.transform_blocks(source, _IDS)

    assert featurizer.output_dim == 3
    assert payloads.shape == (2,)
    np.testing.assert_allclose(payloads[0], _D1)
    np.testing.assert_allclose(payloads[1], _D2)
    assert set(blocks) == {"molgnet_features"}
    assert blocks["molgnet_features"].format is FeatureFormat.RAGGED_SEQUENCE
    assert blocks["molgnet_features"].values.shape == (2,)
    assert MolGNetDrugFeaturizer().fit(source).output_dim == 3


def test_molgnet_reads_uncached_drugs_from_the_source() -> None:
    source = _ragged_source()
    featurizer = MolGNetDrugFeaturizer().fit(source, entity_ids=_IDS[:1])

    payloads = featurizer.transform(source, _IDS[1:])

    # A single tensor makes ``np.array(rows, dtype=object)`` a (1, n, dim) array, so re-cast the row.
    np.testing.assert_allclose(np.asarray(payloads[0], dtype=np.float32), _D2)


def test_molgnet_view_name_and_one_dimensional_tensors() -> None:
    custom = MockFeatureSource(features={"d1": {"custom": _D1}})
    assert MolGNetDrugFeaturizer(view="custom").fit(custom, entity_ids=_IDS[:1]).output_dim == 3
    flat = MockFeatureSource(features={"d1": {"molgnet_features": np.arange(4, dtype=np.float32)}})
    assert MolGNetDrugFeaturizer().fit(flat, entity_ids=_IDS[:1]).output_dim == 4


def test_molgnet_unresolvable_drugs_are_skipped_at_fit_and_raise_at_transform() -> None:
    source = MockFeatureSource(features={"d1": {}, "d2": {}})
    featurizer = MolGNetDrugFeaturizer().fit(source, entity_ids=_IDS[:1])

    assert featurizer.output_dim == 0
    with pytest.raises(KeyError, match="on-the-fly computation failed"):
        featurizer.transform(source, _IDS[:1])
    computed = featurizer._compute_from_source(source, _IDS)
    assert computed.shape[0] == 2
    assert computed[0].shape == (0, 768)
    assert featurizer._compute_single_embedding(source, "d1") is None


def test_molgnet_compute_embedding_failure_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _compute_molgnet_embedding("not-a-smiles") is None
    monkeypatch.setattr(molgnet_module, "get_smiles_for_entities", lambda s, ids: pd.Series({"d1": float("nan")}))
    assert MolGNetDrugFeaturizer()._compute_single_embedding(MockFeatureSource(features={}), "d1") is None


@pytest.mark.parametrize(
    ("blocked", "message"), [("torch", "torch and torch_geometric are required"), ("rdkit", "rdkit is required")]
)
def test_molgnet_compute_embedding_names_the_missing_dependency(
    monkeypatch: pytest.MonkeyPatch, blocked: str, message: str
) -> None:
    block_imports(monkeypatch, blocked)

    with pytest.raises(ImportError, match=message):
        _compute_molgnet_embedding("CCO")


def test_molgnet_computes_missing_drugs_on_the_fly(monkeypatch: pytest.MonkeyPatch, synthetic_dataset) -> None:
    embedding = _patch_embedding(monkeypatch)
    source = _NoStoredTensorSource(synthetic_dataset, synthetic_dataset.drug_ids)
    drug_ids = synthetic_dataset.drug_ids[:2]

    featurizer = MolGNetDrugFeaturizer().fit(source, entity_ids=drug_ids)

    assert featurizer.output_dim == embedding.shape[1]
    assert set(featurizer._features_by_drug) == set(drug_ids)
    payloads = MolGNetDrugFeaturizer().transform(source, drug_ids[:1])
    np.testing.assert_allclose(np.asarray(payloads[0], dtype=np.float32), embedding)
    plain = DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)
    computed = MolGNetDrugFeaturizer()._compute_from_source(plain, drug_ids)
    np.testing.assert_allclose(np.asarray(computed[0], dtype=np.float32), embedding)


@pytest.mark.network
def test_molgnet_computes_an_embedding_from_the_checkpoint() -> None:
    from drevalpy.components.featurizers.drug.molgnet import _get_molgnet_checkpoint

    assert _get_molgnet_checkpoint().endswith("MolGNet.pt")
    embedding = _compute_molgnet_embedding("CCO")
    assert embedding is not None
    assert embedding.shape == (3, 768)


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
    source = _ragged_source()
    featurizer = MolGNetPooledFeaturizer(pooling=pooling).fit(source, entity_ids=_IDS)

    matrix = featurizer.transform(source, _IDS)

    assert featurizer.output_dim == width
    assert matrix.shape == (2, width)
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[1], pool_molgnet(_D2, pooling))
    assert featurizer.transform(source, np.array([], dtype=str)).shape == (0, width)


def test_transform_blocks_emits_one_named_numeric_block() -> None:
    source = _ragged_source()
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
