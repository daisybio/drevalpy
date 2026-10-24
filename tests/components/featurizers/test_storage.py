"""Tests for featurizer variant storage helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.featurizers.storage import (
    VARIANTS_UNS_KEY_DRUG,
    VARIANTS_UNS_KEY_SAMPLE,
    FeaturizerStorageMixin,
    fetch_from_modality,
    fetch_from_obsm,
    fetch_from_varm,
    find_variant_key,
    list_variants,
    make_variant_key,
    next_variant_index,
    register_variant,
)

_SAMPLES = np.array(["cl1", "cl2"])
_DRUGS = np.array(["d1", "d2"])


@pytest.fixture
def mdata() -> md.MuData:
    response = ad.AnnData(
        X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        obs=pd.DataFrame(index=_SAMPLES),
        var=pd.DataFrame(index=_DRUGS),
    )
    response.obsm["pca_expression_0"] = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    response.varm["morgan_fingerprint_0"] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gene_expression = ad.AnnData(
        X=np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32),
        obs=pd.DataFrame(index=_SAMPLES),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(3)]),
    )
    return md.MuData({"response": response, "gene_expression": gene_expression})


def test_variant_registry_round_trip(mdata: md.MuData) -> None:
    assert find_variant_key(mdata, "pca_expression", {"n_components": 2}) is None
    register_variant(mdata, "pca_expression", "pca_expression_0", {"n_components": 2})
    assert find_variant_key(mdata, "pca_expression", {"n_components": 2}) == "pca_expression_0"
    assert find_variant_key(mdata, "pca_expression", {"n_components": 8}) is None
    assert list_variants(mdata, "landmark_genes") == {}
    register_variant(mdata, "chemberta", "chemberta_0", None)
    assert find_variant_key(mdata, "chemberta", None) == "chemberta_0"


def test_registry_is_stored_as_json_per_side(mdata: md.MuData) -> None:
    register_variant(mdata, "smilesvec", "smilesvec_0", {"k": 8}, side="drug")
    assert VARIANTS_UNS_KEY_SAMPLE not in mdata.uns
    assert json.loads(mdata.uns[VARIANTS_UNS_KEY_DRUG]) == {"smilesvec": {"smilesvec_0": {"k": 8}}}
    register_variant(mdata, "shared", "shared_0", {"a": 1}, side="sample")
    register_variant(mdata, "shared", "shared_9", {"a": 1}, side="drug")
    assert find_variant_key(mdata, "shared", {"a": 1}, side="sample") == "shared_0"
    assert find_variant_key(mdata, "shared", {"a": 1}, side="drug") == "shared_9"


def test_list_variants_reads_a_dict_valued_registry(mdata: md.MuData) -> None:
    mdata.uns[VARIANTS_UNS_KEY_SAMPLE] = {"pca": {"pca_0": {"n_components": 2}}}
    assert list_variants(mdata, "pca") == {"pca_0": {"n_components": 2}}


@pytest.mark.parametrize(
    ("storage_key", "index", "expected"),
    [("pca", 0, "pca_0"), ("raw[gene_expression]", 1, "raw_gene_expression_1"), ("a:b", 2, "a_b_2")],
)
def test_make_variant_key_sanitizes_the_storage_key(storage_key: str, index: int, expected: str) -> None:
    assert make_variant_key(storage_key, index) == expected


def test_next_variant_index_counts_registered_variants(mdata: md.MuData) -> None:
    assert next_variant_index(mdata, "pca") == 0
    register_variant(mdata, "pca", "pca_0", {"n_components": 2})
    register_variant(mdata, "pca", "pca_1", {"n_components": 8})
    assert next_variant_index(mdata, "pca") == 2


@pytest.mark.parametrize(
    ("fetch", "key", "present", "expected"),
    [
        (fetch_from_modality, "gene_expression", "methylation", [[8.0, 9.0, 10.0], [5.0, 6.0, 7.0]]),
        (fetch_from_varm, "morgan_fingerprint_0", "chemberta_0", [[0.0, 1.0], [1.0, 0.0]]),
        (fetch_from_obsm, "pca_expression_0", "bionic_0", [[0.3, 0.4], [0.1, 0.2]]),
    ],
)
def test_fetchers_align_rows_and_report_absence(mdata: md.MuData, fetch, key, present, expected) -> None:
    ids = _DRUGS if fetch is fetch_from_varm else _SAMPLES
    result = fetch(mdata, key, ids[::-1])
    assert result is not None
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    assert fetch(mdata, present, ids) is None
    padded = fetch(mdata, key, np.array([ids[0], "ghost"]))
    assert padded is not None
    assert np.all(np.isnan(padded[1])) and not np.isnan(padded[0]).any()


def test_fetchers_return_none_without_a_response_modality() -> None:
    omics = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32), obs=pd.DataFrame(index=_SAMPLES))
    only_omics = md.MuData({"gene_expression": omics})
    assert fetch_from_varm(only_omics, "anything", _DRUGS) is None
    assert fetch_from_obsm(only_omics, "anything", _SAMPLES) is None


class _SampleStore(FeaturizerStorageMixin):
    storage_key = "probe"
    side = "sample"


class _DrugStore(FeaturizerStorageMixin):
    storage_key = "probe"
    side = "drug"


def test_featurizer_inherits_the_storage_mixin() -> None:
    assert issubclass(Featurizer, FeaturizerStorageMixin)
    for name in ("fetch", "_fetch_by_key", "store", "_store_by_key", "list_stored_variants"):
        assert name not in vars(Featurizer), f"{name} should live only on the mixin"


def test_storage_mixin_sample_side_store_fetch_round_trip(mdata: md.MuData) -> None:
    values = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    store = _SampleStore()
    assert store.fetch(mdata, _SAMPLES) is None
    store.store(mdata, _SAMPLES, values, {"n": 3})
    store.store(mdata, _SAMPLES, values, {"n": 4})
    np.testing.assert_allclose(store.fetch(mdata, _SAMPLES, {"n": 3}), values)
    assert _SampleStore.list_stored_variants(mdata) == {"probe_0": {"n": 3}, "probe_1": {"n": 4}}
    assert _DrugStore.list_stored_variants(mdata) == {}


def test_storage_mixin_drug_side_writes_to_varm(mdata: md.MuData) -> None:
    _DrugStore().store(mdata, _DRUGS, np.eye(2, dtype=np.float32))
    assert "probe_0" in mdata.mod["response"].varm
    assert "probe_0" not in mdata.mod["response"].obsm


def test_fetch_precomputed_reads_through_a_source(mdata: md.MuData) -> None:
    values = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
    store = _SampleStore()
    store.store(mdata, _SAMPLES, values)
    source = SimpleNamespace(mdata=mdata)
    np.testing.assert_allclose(store.fetch_precomputed(source, _SAMPLES), values)
    assert store.fetch_precomputed(SimpleNamespace(mdata=None), _SAMPLES) is None


def test_storage_mixin_prefers_a_modality_over_obsm(mdata: md.MuData) -> None:
    register_variant(mdata, "probe", "gene_expression", None, side="sample")
    fetched = _SampleStore().fetch(mdata, _SAMPLES)
    assert fetched is not None
    assert fetched.shape == (2, 3)
