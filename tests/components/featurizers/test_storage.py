"""Tests for featurizer variant storage helpers.

Mirrors :mod:`drevalpy.components.featurizers.storage`. These are pure functions
over a MuData object; the only production writer is ``Dataset.precompute()``, so
the fixtures below build the 2x2 MuData by hand.
"""

from __future__ import annotations

import json

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from drevalpy.components.featurizers.storage import (
    VARIANTS_UNS_KEY_CELL_LINE,
    VARIANTS_UNS_KEY_DRUG,
    fetch_from_modality,
    fetch_from_obsm,
    fetch_from_varm,
    find_variant_key,
    list_variants,
    make_variant_key,
    next_variant_index,
    register_variant,
)

_CELL_LINES = np.array(["cl1", "cl2"])
_DRUGS = np.array(["d1", "d2"])


@pytest.fixture
def mdata() -> md.MuData:
    """A 2x2 response MuData with one omics modality and pre-computed obsm/varm."""
    response = ad.AnnData(
        X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        obs=pd.DataFrame(index=_CELL_LINES),
        var=pd.DataFrame(index=_DRUGS),
    )
    response.obsm["pca_expression_0"] = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    response.varm["morgan_fingerprint_0"] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gene_expression = ad.AnnData(
        X=np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32),
        obs=pd.DataFrame(index=_CELL_LINES),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(3)]),
    )
    return md.MuData({"response": response, "gene_expression": gene_expression})


def test_find_variant_key_returns_none_without_a_registry(mdata: md.MuData) -> None:
    assert find_variant_key(mdata, "pca_expression", {"n_components": 2}) is None


def test_register_then_find_matches_on_hyperparameters(mdata: md.MuData) -> None:
    register_variant(mdata, "pca_expression", "pca_expression_0", {"n_components": 2})

    assert find_variant_key(mdata, "pca_expression", {"n_components": 2}) == "pca_expression_0"


def test_find_variant_key_returns_none_for_unmatched_hyperparameters(mdata: md.MuData) -> None:
    register_variant(mdata, "pca_expression", "pca_expression_0", {"n_components": 2})

    assert find_variant_key(mdata, "pca_expression", {"n_components": 8}) is None


def test_find_variant_key_treats_none_as_the_default_empty_params(mdata: md.MuData) -> None:
    register_variant(mdata, "chemberta", "chemberta_0", None)

    assert find_variant_key(mdata, "chemberta", None) == "chemberta_0"


def test_register_variant_writes_json_under_the_side_specific_uns_key(mdata: md.MuData) -> None:
    register_variant(mdata, "smilesvec", "smilesvec_0", {"k": 8}, side="drug")

    assert VARIANTS_UNS_KEY_CELL_LINE not in mdata.uns
    assert json.loads(mdata.uns[VARIANTS_UNS_KEY_DRUG]) == {"smilesvec": {"smilesvec_0": {"k": 8}}}


def test_registry_sides_are_independent(mdata: md.MuData) -> None:
    register_variant(mdata, "shared", "shared_0", {"a": 1}, side="cell_line")
    register_variant(mdata, "shared", "shared_9", {"a": 1}, side="drug")

    assert find_variant_key(mdata, "shared", {"a": 1}, side="cell_line") == "shared_0"
    assert find_variant_key(mdata, "shared", {"a": 1}, side="drug") == "shared_9"


def test_list_variants_is_empty_for_an_unknown_storage_key(mdata: md.MuData) -> None:
    register_variant(mdata, "pca_expression", "pca_expression_0", {"n_components": 2})

    assert list_variants(mdata, "landmark_genes") == {}


def test_list_variants_reads_a_dict_valued_registry(mdata: md.MuData) -> None:
    mdata.uns[VARIANTS_UNS_KEY_CELL_LINE] = {"pca": {"pca_0": {"n_components": 2}}}

    assert list_variants(mdata, "pca") == {"pca_0": {"n_components": 2}}


@pytest.mark.parametrize(
    ("storage_key", "index", "expected"),
    [
        pytest.param("pca", 0, "pca_0", id="plain"),
        pytest.param("raw[gene_expression]", 1, "raw_gene_expression_1", id="bracketed-view"),
        pytest.param("a:b", 2, "a_b_2", id="colon"),
    ],
)
def test_make_variant_key_sanitizes_the_storage_key(storage_key: str, index: int, expected: str) -> None:
    assert make_variant_key(storage_key, index) == expected


def test_next_variant_index_counts_registered_variants(mdata: md.MuData) -> None:
    assert next_variant_index(mdata, "pca") == 0

    register_variant(mdata, "pca", "pca_0", {"n_components": 2})
    register_variant(mdata, "pca", "pca_1", {"n_components": 8})

    assert next_variant_index(mdata, "pca") == 2


def test_fetch_from_modality_aligns_rows_to_entity_ids(mdata: md.MuData) -> None:
    result = fetch_from_modality(mdata, "gene_expression", np.array(["cl2", "cl1"]))

    assert result is not None
    np.testing.assert_allclose(result, [[8.0, 9.0, 10.0], [5.0, 6.0, 7.0]])


def test_fetch_from_modality_returns_none_for_an_absent_modality(mdata: md.MuData) -> None:
    assert fetch_from_modality(mdata, "methylation", _CELL_LINES) is None


def test_fetch_from_modality_fills_unknown_entities_with_nan(mdata: md.MuData) -> None:
    result = fetch_from_modality(mdata, "gene_expression", np.array(["cl1", "ghost"]))

    assert result is not None
    np.testing.assert_allclose(result[0], [5.0, 6.0, 7.0])
    assert np.all(np.isnan(result[1]))


def test_fetch_from_varm_aligns_rows_to_drug_ids(mdata: md.MuData) -> None:
    result = fetch_from_varm(mdata, "morgan_fingerprint_0", np.array(["d2", "d1"]))

    assert result is not None
    np.testing.assert_allclose(result, [[0.0, 1.0], [1.0, 0.0]])


def test_fetch_from_varm_returns_none_for_an_absent_key(mdata: md.MuData) -> None:
    assert fetch_from_varm(mdata, "chemberta_0", _DRUGS) is None


def test_fetch_from_varm_fills_unknown_drugs_with_nan(mdata: md.MuData) -> None:
    result = fetch_from_varm(mdata, "morgan_fingerprint_0", np.array(["d1", "ghost"]))

    assert result is not None
    np.testing.assert_allclose(result[0], [1.0, 0.0])
    assert np.all(np.isnan(result[1]))


def test_fetch_from_obsm_aligns_rows_to_cell_line_ids(mdata: md.MuData) -> None:
    result = fetch_from_obsm(mdata, "pca_expression_0", np.array(["cl2", "cl1"]))

    assert result is not None
    np.testing.assert_allclose(result, [[0.3, 0.4], [0.1, 0.2]], rtol=1e-6)


def test_fetch_from_obsm_returns_none_for_an_absent_key(mdata: md.MuData) -> None:
    assert fetch_from_obsm(mdata, "bionic_0", _CELL_LINES) is None


def test_fetch_from_obsm_fills_unknown_cell_lines_with_nan(mdata: md.MuData) -> None:
    result = fetch_from_obsm(mdata, "pca_expression_0", np.array(["cl1", "ghost"]))

    assert result is not None
    np.testing.assert_allclose(result[0], [0.1, 0.2], rtol=1e-6)
    assert np.all(np.isnan(result[1]))


def test_fetchers_return_none_without_a_response_modality() -> None:
    only_omics = md.MuData(
        {
            "gene_expression": ad.AnnData(
                X=np.zeros((2, 2), dtype=np.float32),
                obs=pd.DataFrame(index=_CELL_LINES),
                var=pd.DataFrame(index=["g0", "g1"]),
            )
        }
    )

    assert fetch_from_varm(only_omics, "anything", _DRUGS) is None
    assert fetch_from_obsm(only_omics, "anything", _CELL_LINES) is None
