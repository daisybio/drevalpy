"""Tests for the ``FeatureSource`` ABC and its two Dataset-backed adapters.

The special cases each adapter's ``get_entity_view`` carries are the point of
this module: ``"tissue"`` is metadata rather than a matrix on the cell-line side,
and ``"drug_graph"`` is a ``uns`` dict rather than a ``varm`` row on the drug side.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import (
    CellLineFeatureSource,
    DrugFeatureSource,
    FeatureSource,
)
from tests.synthetic import CHEMBERTA_DIM, N_GENES


@pytest.fixture()
def cell_line_source(synthetic_dataset: Dataset) -> CellLineFeatureSource:
    """Cell-line adapter over the first five cell lines."""
    return CellLineFeatureSource(synthetic_dataset, synthetic_dataset.cell_line_ids[:5])


@pytest.fixture()
def drug_source(synthetic_dataset: Dataset) -> DrugFeatureSource:
    """Drug adapter over every drug in the fixture."""
    return DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)


class TestSharedBase:
    def test_feature_source_cannot_be_instantiated(self, synthetic_dataset: Dataset):
        with pytest.raises(TypeError, match="abstract"):
            FeatureSource(synthetic_dataset, np.array(["x"]))  # type: ignore[abstract]

    def test_identifiers_are_coerced_to_strings(self, synthetic_dataset: Dataset):
        source = CellLineFeatureSource(synthetic_dataset, np.array([1, 2]))

        assert source.identifiers.dtype.kind == "U"
        assert source.identifiers.tolist() == ["1", "2"]

    def test_identifiers_preserve_the_requested_order(self, synthetic_dataset: Dataset):
        ids = synthetic_dataset.cell_line_ids[[3, 0, 1]]

        source = CellLineFeatureSource(synthetic_dataset, ids)

        assert source.identifiers.tolist() == list(ids)

    def test_mdata_exposes_the_dataset_backing_object(
        self, synthetic_dataset: Dataset, cell_line_source: CellLineFeatureSource
    ):
        assert cell_line_source.mdata is synthetic_dataset.mdata

    def test_get_metadata_reads_dataset_uns(self, cell_line_source: CellLineFeatureSource):
        assert set(cell_line_source.get_metadata("sparsego")) == {"gene2ind", "ontology"}

    def test_get_metadata_propagates_missing_keys(self, cell_line_source: CellLineFeatureSource):
        with pytest.raises(KeyError, match="nonexistent"):
            cell_line_source.get_metadata("nonexistent")


class TestCellLineFeatureSource:
    def test_get_view_matrix_returns_one_row_per_requested_id(self, cell_line_source: CellLineFeatureSource):
        matrix = cell_line_source.get_view_matrix("gene_expression", cell_line_source.identifiers)

        assert matrix.shape == (5, N_GENES)

    def test_get_feature_names_returns_gene_symbols(self, cell_line_source: CellLineFeatureSource):
        names = cell_line_source.get_feature_names("gene_expression")

        assert names is not None
        assert len(names) == N_GENES

    def test_get_feature_names_is_none_for_an_unknown_view(self, cell_line_source: CellLineFeatureSource):
        assert cell_line_source.get_feature_names("absent_view") is None

    def test_get_entity_view_special_cases_tissue(
        self, synthetic_dataset: Dataset, cell_line_source: CellLineFeatureSource
    ):
        entity_id = str(cell_line_source.identifiers[0])

        tissue = cell_line_source.get_entity_view(entity_id, "tissue")

        assert tissue == synthetic_dataset.get_tissue(np.array([entity_id]))[0]

    def test_tissue_of_an_unknown_cell_line_is_nan(self, cell_line_source: CellLineFeatureSource):
        assert np.isnan(cell_line_source.get_entity_view("NOT_A_CELL_LINE", "tissue"))

    def test_get_entity_view_returns_a_single_omics_row(self, cell_line_source: CellLineFeatureSource):
        entity_id = str(cell_line_source.identifiers[0])

        row = cell_line_source.get_entity_view(entity_id, "gene_expression")

        assert row.shape == (N_GENES,)

    def test_get_entity_view_row_matches_the_matrix_row(self, cell_line_source: CellLineFeatureSource):
        entity_id = str(cell_line_source.identifiers[2])

        row = cell_line_source.get_entity_view(entity_id, "gene_expression")

        matrix = cell_line_source.get_view_matrix("gene_expression", np.array([entity_id]))
        np.testing.assert_array_equal(row, matrix[0])


class TestDrugFeatureSource:
    def test_get_view_matrix_returns_one_row_per_requested_id(self, drug_source: DrugFeatureSource):
        matrix = drug_source.get_view_matrix("chemberta", drug_source.identifiers[:3])

        assert matrix.shape == (3, CHEMBERTA_DIM)

    def test_get_feature_names_covers_every_varm_column(self, drug_source: DrugFeatureSource):
        names = drug_source.get_feature_names("chemberta")

        assert names is not None
        assert len(names) == CHEMBERTA_DIM

    def test_get_feature_names_is_none_for_an_unknown_view(self, drug_source: DrugFeatureSource):
        assert drug_source.get_feature_names("absent_view") is None

    def test_get_entity_view_special_cases_drug_graph(self, drug_source: DrugFeatureSource):
        graph = drug_source.get_entity_view(str(drug_source.identifiers[0]), "drug_graph")

        assert graph is not None
        assert set(graph) >= {"x", "edge_index", "edge_attr"}

    def test_graph_of_an_unknown_drug_is_none(self, drug_source: DrugFeatureSource):
        assert drug_source.get_entity_view("NOT_A_DRUG", "drug_graph") is None

    def test_get_entity_view_returns_a_single_varm_row(self, drug_source: DrugFeatureSource):
        row = drug_source.get_entity_view(str(drug_source.identifiers[0]), "chemberta")

        assert row.shape == (CHEMBERTA_DIM,)
