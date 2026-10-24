"""Tests for the MolGNet ragged drug featurizer; the checkpoint load is the only network-bound part."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import drevalpy.components.featurizers.drug.molgnet as molgnet_module
from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.drug.molgnet import (
    MolGNetDrugFeaturizer,
    _compute_molgnet_embedding,
)
from drevalpy.types.data.feature_source import DrugFeatureSource
from tests._import_shims import block_imports
from tests.conftest import MockFeatureSource

_D1 = np.arange(6, dtype=np.float32).reshape(2, 3)
_D2 = np.arange(9, dtype=np.float32).reshape(3, 3)
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
