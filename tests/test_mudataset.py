"""Tests for MuDataset against the TOYv2.h5mu fixture."""

# ruff: noqa: D102

import numpy as np
import pytest

from drevalpy.data.mudataset import MuDataset

DATA_PATH = "data/TOYv2.h5mu"


@pytest.fixture()
def mudataset() -> MuDataset:
    """Load the TOYv2 fixture."""
    return MuDataset.from_file(DATA_PATH)


class TestFromFile:
    """Test loading and basic properties."""

    def test_loads_without_error(self, mudataset: MuDataset):
        assert mudataset is not None

    def test_repr(self, mudataset: MuDataset):
        r = repr(mudataset)
        assert "MuDataset" in r
        assert "cell_lines=" in r

    def test_cell_line_ids(self, mudataset: MuDataset):
        ids = mudataset.cell_line_ids
        assert ids.ndim == 1
        assert len(ids) == 90
        assert ids.dtype.kind in ("U", "O")

    def test_drug_ids(self, mudataset: MuDataset):
        ids = mudataset.drug_ids
        assert ids.ndim == 1
        assert len(ids) == 36
        assert ids.dtype.kind in ("U", "O")


class TestResponse:
    """Test response matrix access."""

    def test_response_matrix_shape(self, mudataset: MuDataset):
        mat = mudataset.response_matrix
        assert mat.shape == (90, 36)
        assert mat.dtype == np.float32

    def test_response_layer_auc(self, mudataset: MuDataset):
        auc = mudataset.get_response_layer("AUC")
        assert auc.shape == (90, 36)
        assert auc.dtype == np.float32

    def test_response_layer_missing(self, mudataset: MuDataset):
        with pytest.raises(KeyError, match="nonexistent"):
            mudataset.get_response_layer("nonexistent")


class TestCellLineFeatures:
    """Test cell-line feature retrieval."""

    def test_gene_expression(self, mudataset: MuDataset):
        ids = mudataset.cell_line_ids[:5]
        features = mudataset.get_cell_line_features("gene_expression", ids)
        assert features.shape[0] == 5
        assert features.shape[1] == 13
        assert features.dtype == np.float32

    def test_missing_ids_get_nan(self, mudataset: MuDataset):
        ids = np.array(["FAKE_ID_1", "FAKE_ID_2"])
        features = mudataset.get_cell_line_features("gene_expression", ids)
        assert features.shape == (2, 13)
        assert np.all(np.isnan(features))

    def test_pathway_features(self, mudataset: MuDataset):
        ids = mudataset.cell_line_ids[:3]
        features = mudataset.get_cell_line_features("pathway_features", ids)
        assert features.shape[0] == 3
        assert features.dtype == np.float32

    def test_unknown_modality_raises(self, mudataset: MuDataset):
        with pytest.raises(KeyError, match="nonexistent"):
            mudataset.get_cell_line_features("nonexistent", mudataset.cell_line_ids[:1])


class TestDrugFeatures:
    """Test drug feature retrieval."""

    def test_chemberta(self, mudataset: MuDataset):
        ids = mudataset.drug_ids[:4]
        features = mudataset.get_drug_features("chemberta", ids)
        assert features.shape[0] == 4
        assert features.dtype == np.float32

    def test_morgan_fingerprint(self, mudataset: MuDataset):
        ids = mudataset.drug_ids
        features = mudataset.get_drug_features("morgan_fingerprint", ids)
        assert features.shape == (36, 128)

    def test_missing_drug_raises(self, mudataset: MuDataset):
        with pytest.raises(KeyError, match="nonexistent"):
            mudataset.get_drug_features("nonexistent", mudataset.drug_ids[:1])


class TestDrugGraphs:
    """Test drug graph access."""

    def test_get_drug_graphs(self, mudataset: MuDataset):
        ids = mudataset.drug_ids[:3]
        graphs = mudataset.get_drug_graphs(ids)
        assert len(graphs) == 3
        for g in graphs:
            if g is not None:
                assert "x" in g
                assert "edge_index" in g
                assert "edge_attr" in g


class TestMetadata:
    """Test metadata access."""

    def test_cell_line_meta(self, mudataset: MuDataset):
        meta = mudataset.cell_line_meta
        assert "cell_line_name" in meta.columns or "tissue" in meta.columns

    def test_get_tissue(self, mudataset: MuDataset):
        ids = mudataset.cell_line_ids[:5]
        tissues = mudataset.get_tissue(ids)
        assert len(tissues) == 5

    def test_get_tissue_unknown_id(self, mudataset: MuDataset):
        tissues = mudataset.get_tissue(np.array(["FAKE_ID"]))
        assert len(tissues) == 1


class TestSubsetting:
    """Test subsetting operations."""

    def test_subset_cell_lines(self, mudataset: MuDataset):
        ids = mudataset.cell_line_ids[:10]
        sub = mudataset.subset_cell_lines(ids)
        assert len(sub.cell_line_ids) == 10
        assert sub.response_matrix.shape == (10, 36)

    def test_subset_drugs(self, mudataset: MuDataset):
        ids = mudataset.drug_ids[:5]
        sub = mudataset.subset_drugs(ids)
        assert len(sub.drug_ids) == 5
        assert sub.response_matrix.shape == (90, 5)

    def test_subset_preserves_uns(self, mudataset: MuDataset):
        sub = mudataset.subset_cell_lines(mudataset.cell_line_ids[:5])
        assert "drug_graphs" in sub.mdata.uns


class TestUns:
    """Test uns access."""

    def test_get_uns(self, mudataset: MuDataset):
        bpe = mudataset.get_uns("bpe_codes")
        assert isinstance(bpe, str)

    def test_get_uns_missing(self, mudataset: MuDataset):
        with pytest.raises(KeyError, match="nonexistent"):
            mudataset.get_uns("nonexistent")
