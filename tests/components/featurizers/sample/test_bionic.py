"""Tests for the BIONIC PPI sample featurizer; only ``_load_ppi_data`` needs the artifact download."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.sample.bionic import (
    BionicSampleFeaturizer,
    _aggregate_ppi_for_sample,
)
from tests.conftest import MockFeatureSource

_GENE_NAMES = ("g_high", "g_mid", "g_low")
_EXPR_ROW = np.array([3.0, 2.0, 1.0])
_PPI_LOOKUP = {
    "g_high": np.array([1.0, 5.0], dtype=np.float32),
    "g_mid": np.array([3.0, 1.0], dtype=np.float32),
    "g_low": np.array([100.0, 100.0], dtype=np.float32),
}


def test_hyperparameter_space_and_default_gene_add_num() -> None:
    assert set(BionicSampleFeaturizer.get_hyperparameter_space()) == {"gene_add_num", "aggregation"}
    assert BionicSampleFeaturizer()._gene_add_num == 256


def _aggregate(expr, names, eligible, lookup=_PPI_LOOKUP, *, gene_add_num=2, embed_dim=2, aggregation="mean"):
    return _aggregate_ppi_for_sample(
        expr,
        names,
        eligible,
        lookup,
        gene_add_num=gene_add_num,
        embed_dim=embed_dim,
        aggregation=aggregation,
    )


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [("mean", [2.0, 3.0]), ("max", [3.0, 5.0]), ("sum", [4.0, 6.0]), ("unrecognised", [2.0, 3.0])],
)
def test_aggregate_selects_the_top_expressed_eligible_genes(aggregation: str, expected: list[float]) -> None:
    vector = _aggregate(_EXPR_ROW, _GENE_NAMES, {"g_high", "g_mid"}, aggregation=aggregation)

    np.testing.assert_allclose(vector, expected)
    assert vector.dtype == np.float32


def test_aggregate_budget_ties_and_empty_eligibility() -> None:
    np.testing.assert_allclose(_aggregate(_EXPR_ROW, _GENE_NAMES, set(_GENE_NAMES), gene_add_num=1), [1.0, 5.0])
    np.testing.assert_allclose(_aggregate(_EXPR_ROW, _GENE_NAMES, set()), [0.0, 0.0])
    lookup = {name: np.array([float(i)], dtype=np.float32) for i, name in enumerate("abcd")}
    tie = _aggregate(np.zeros(4), tuple("abcd"), set("abcd"), lookup, embed_dim=1)
    np.testing.assert_allclose(tie, [0.5])


def test_bionic_reads_the_bionic_features_view_when_present() -> None:
    source = MockFeatureSource(
        features={"cl1": {"bionic_features": np.array([0.1, 0.2])}, "cl2": {"bionic_features": np.array([0.3, 0.4])}}
    )
    ids = np.array(["cl1", "cl2"], dtype=str)

    featurizer = BionicSampleFeaturizer().fit(source, entity_ids=ids)

    assert featurizer.output_dim == 2
    np.testing.assert_allclose(featurizer.transform(source, ids), [[0.1, 0.2], [0.3, 0.4]], rtol=1e-6)


def test_bionic_compute_from_source_aggregates_per_sample_and_needs_feature_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drevalpy.components.featurizers.sample.bionic as bionic_module

    ppi = (np.array([[1.0, 5.0], [3.0, 1.0]], dtype=np.float32), ["g_high", "g_mid"], {"g_high", "g_mid"})
    monkeypatch.setattr(bionic_module, "_load_ppi_data", lambda: ppi)
    named = MockFeatureSource(
        features={"cl1": {"gene_expression": np.array([3.0, 2.0])}, "cl2": {"gene_expression": np.array([1.0, 4.0])}},
        meta_info={"gene_expression": ["g_high", "g_mid"]},
    )

    matrix = BionicSampleFeaturizer()._compute_from_source(named, np.array(["cl1", "cl2"], dtype=str))

    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix[0], [2.0, 3.0])
    nameless = MockFeatureSource(features={"cl1": {"gene_expression": np.array([1.0, 2.0])}})
    with pytest.raises(ValueError, match="must provide feature names"):
        BionicSampleFeaturizer()._compute_from_source(nameless, np.array(["cl1"], dtype=str))


@pytest.mark.network
def test_load_ppi_data_returns_features_gene_names_and_selection() -> None:
    from drevalpy.components.featurizers.sample.bionic import _load_ppi_data

    ppi_features, ppi_gene_names, gene_list_sel = _load_ppi_data()

    assert ppi_features.ndim == 2
    assert len(ppi_gene_names) == ppi_features.shape[0]
    assert gene_list_sel
