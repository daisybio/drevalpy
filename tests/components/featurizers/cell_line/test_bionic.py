"""Tests for the BIONIC PPI cell-line featurizer.

Mirrors :mod:`drevalpy.components.featurizers.cell_line.bionic`. Only
``_load_ppi_data`` needs the 83 MB artifact download, so the aggregation kernel is
tested directly with a hand-built PPI lookup.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.cell_line.bionic import (
    BionicCellLineFeaturizer,
    _aggregate_ppi_for_cell_line,
)
from tests.conftest import MockFeatureSource

_GENE_NAMES = ("g_high", "g_mid", "g_low")
_EXPR_ROW = np.array([3.0, 2.0, 1.0])
_PPI_LOOKUP = {
    "g_high": np.array([1.0, 5.0], dtype=np.float32),
    "g_mid": np.array([3.0, 1.0], dtype=np.float32),
    "g_low": np.array([100.0, 100.0], dtype=np.float32),
}


def test_hyperparameter_space_exposes_gene_add_num_and_aggregation() -> None:
    assert set(BionicCellLineFeaturizer.get_hyperparameter_space()) == {"gene_add_num", "aggregation"}


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        pytest.param("mean", [2.0, 3.0], id="mean"),
        pytest.param("max", [3.0, 5.0], id="max"),
        pytest.param("sum", [4.0, 6.0], id="sum"),
        pytest.param("unrecognised", [2.0, 3.0], id="unknown-falls-back-to-mean"),
    ],
)
def test_aggregate_selects_the_top_expressed_eligible_genes(
    aggregation: str,
    expected: list[float],
) -> None:
    vector = _aggregate_ppi_for_cell_line(
        _EXPR_ROW,
        _GENE_NAMES,
        {"g_high", "g_mid"},
        _PPI_LOOKUP,
        gene_add_num=2,
        embed_dim=2,
        aggregation=aggregation,
    )

    np.testing.assert_allclose(vector, expected)
    assert vector.dtype == np.float32


def test_aggregate_honours_the_gene_add_num_budget() -> None:
    vector = _aggregate_ppi_for_cell_line(
        _EXPR_ROW,
        _GENE_NAMES,
        set(_GENE_NAMES),
        _PPI_LOOKUP,
        gene_add_num=1,
        embed_dim=2,
        aggregation="mean",
    )

    np.testing.assert_allclose(vector, [1.0, 5.0])


def test_aggregate_returns_a_zero_vector_when_nothing_is_eligible() -> None:
    vector = _aggregate_ppi_for_cell_line(
        _EXPR_ROW,
        _GENE_NAMES,
        set(),
        _PPI_LOOKUP,
        gene_add_num=2,
        embed_dim=2,
        aggregation="mean",
    )

    np.testing.assert_allclose(vector, [0.0, 0.0])


def test_bionic_reads_the_bionic_features_view_when_present() -> None:
    source = MockFeatureSource(
        features={
            "cl1": {"bionic_features": np.array([0.1, 0.2])},
            "cl2": {"bionic_features": np.array([0.3, 0.4])},
        }
    )
    ids = np.array(["cl1", "cl2"], dtype=str)

    featurizer = BionicCellLineFeaturizer().fit(source, entity_ids=ids)

    assert featurizer.output_dim == 2
    np.testing.assert_allclose(featurizer.transform(source, ids), [[0.1, 0.2], [0.3, 0.4]], rtol=1e-6)


def test_bionic_compute_from_source_requires_gene_expression_feature_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drevalpy.components.featurizers.cell_line.bionic as bionic_module

    monkeypatch.setattr(
        bionic_module,
        "_load_ppi_data",
        lambda: (np.zeros((1, 2), dtype=np.float32), ["g_high"], {"g_high"}),
    )
    source = MockFeatureSource(features={"cl1": {"gene_expression": np.array([1.0, 2.0])}})

    with pytest.raises(ValueError, match="must provide feature names"):
        BionicCellLineFeaturizer()._compute_from_source(source, np.array(["cl1"], dtype=str))


def test_bionic_compute_from_source_aggregates_per_cell_line(monkeypatch: pytest.MonkeyPatch) -> None:
    import drevalpy.components.featurizers.cell_line.bionic as bionic_module

    monkeypatch.setattr(
        bionic_module,
        "_load_ppi_data",
        lambda: (np.array([[1.0, 5.0], [3.0, 1.0]], dtype=np.float32), ["g_high", "g_mid"], {"g_high", "g_mid"}),
    )
    source = MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([3.0, 2.0])},
            "cl2": {"gene_expression": np.array([1.0, 4.0])},
        },
        meta_info={"gene_expression": ["g_high", "g_mid"]},
    )

    matrix = BionicCellLineFeaturizer()._compute_from_source(source, np.array(["cl1", "cl2"], dtype=str))

    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix[0], [2.0, 3.0])


@pytest.mark.network
def test_load_ppi_data_returns_features_gene_names_and_selection() -> None:
    from drevalpy.components.featurizers.cell_line.bionic import _load_ppi_data

    ppi_features, ppi_gene_names, gene_list_sel = _load_ppi_data()

    assert ppi_features.ndim == 2
    assert len(ppi_gene_names) == ppi_features.shape[0]
    assert gene_list_sel
