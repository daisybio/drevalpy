"""Tests for ``Dataset`` against a synthetic dataset written to and read from disk.

This module used to load a gitignored, downloaded ``.h5mu`` from the repo-local
data directory, which meant it could not run on a clean checkout. It now writes
the synthetic fixture to ``tmp_path`` and reads it back, so the one thing an
in-memory fixture would otherwise drop -- real on-disk ``.h5mu`` I/O -- stays
covered. Expected dimensions are derived from the builder's constants rather
than hard-coded, so the fixture and the assertions cannot drift apart.
"""

# ruff: noqa: D102

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.types.data.dataset import Dataset
from tests.synthetic import (
    BPE_LENGTH,
    BUILTIN_MEASURE,
    CHEMBERTA_DIM,
    CNV_MODALITY,
    FINGERPRINT_BITS,
    N_CELL_LINES,
    N_DRUGS,
    N_GENES,
    N_PATHWAYS,
    OMICS_MODALITIES,
    build_synthetic_dataset,
)
from tests.types.data._helpers import stub_featurizer, stub_fit_transform_featurizer


@pytest.fixture(scope="module")
def h5mu_path(tmp_path_factory: pytest.TempPathFactory) -> UPath:
    """Write the synthetic dataset to a real .h5mu file once per module."""
    path = UPath(tmp_path_factory.mktemp("dataset")) / "synthetic.h5mu"
    build_synthetic_dataset().save(path)
    return path


@pytest.fixture()
def mudataset(h5mu_path: UPath) -> Dataset:
    """Read the synthetic dataset back from disk."""
    return Dataset.load(h5mu_path)


class TestRoundTrip:
    """Test that a saved dataset reads back with its identity and structure intact."""

    def test_loads_without_error(self, mudataset: Dataset):
        assert mudataset is not None

    def test_name_survives_the_round_trip(self, mudataset: Dataset):
        assert mudataset.name == build_synthetic_dataset().name

    def test_all_modalities_survive_the_round_trip(self, mudataset: Dataset):
        assert set(mudataset.mdata.mod) == {"response", *OMICS_MODALITIES}

    def test_copy_number_is_stored_under_the_accessor_name(self, mudataset: Dataset):
        """The datasets store CNV without the ``_gistic`` suffix; the fixture follows suit."""
        assert CNV_MODALITY in mudataset.mdata.mod
        assert CNV_MODALITY == "copy_number_variation"

    def test_repr(self, mudataset: Dataset):
        r = repr(mudataset)
        assert "Dataset" in r
        assert "Cell lines:" in r

    def test_cell_line_ids(self, mudataset: Dataset):
        ids = mudataset.cell_line_ids
        assert ids.ndim == 1
        assert len(ids) == N_CELL_LINES
        assert ids.dtype.kind in ("U", "O")

    def test_drug_ids(self, mudataset: Dataset):
        ids = mudataset.drug_ids
        assert ids.ndim == 1
        assert len(ids) == N_DRUGS
        assert ids.dtype.kind in ("U", "O")


class TestResponse:
    """Test response matrix access."""

    def test_response_matrix_shape(self, mudataset: Dataset):
        mat = mudataset.response_matrix
        assert mat.shape == (N_CELL_LINES, N_DRUGS)
        assert mat.dtype == np.float32

    def test_response_matrix_is_nan_sparse(self, mudataset: Dataset):
        mat = mudataset.response_matrix
        assert np.isnan(mat).any()
        assert not np.isnan(mat).all()

    def test_response_layer_auc(self, mudataset: Dataset):
        auc = mudataset.get_response_layer("AUC")
        assert auc.shape == (N_CELL_LINES, N_DRUGS)
        assert auc.dtype == np.float32

    def test_builtin_measure_layer_matches_x(self, mudataset: Dataset):
        layer = mudataset.get_response_layer(BUILTIN_MEASURE)
        np.testing.assert_array_equal(np.isnan(layer), np.isnan(mudataset.response_matrix))

    def test_response_layer_missing(self, mudataset: Dataset):
        with pytest.raises(KeyError, match="nonexistent"):
            mudataset.get_response_layer("nonexistent")


class TestCellLineFeatures:
    """Test cell-line feature retrieval."""

    def test_gene_expression(self, mudataset: Dataset):
        ids = mudataset.cell_line_ids[:5]
        features = mudataset.get_cell_line_features("gene_expression", ids)
        assert features.shape == (5, N_GENES)
        assert features.dtype == np.float32

    def test_missing_ids_get_nan(self, mudataset: Dataset):
        ids = np.array(["FAKE_ID_1", "FAKE_ID_2"])
        features = mudataset.get_cell_line_features("gene_expression", ids)
        assert features.shape == (2, N_GENES)
        assert np.all(np.isnan(features))

    def test_pathway_features(self, mudataset: Dataset):
        ids = mudataset.cell_line_ids[:3]
        features = mudataset.get_cell_line_features("pathway_features", ids)
        assert features.shape == (3, N_PATHWAYS)
        assert features.dtype == np.float32

    def test_gene_names_are_real_symbols(self, mudataset: Dataset):
        names = mudataset.get_cell_line_feature_names("gene_expression")
        assert names is not None
        assert len(names) == N_GENES
        assert all(name.isupper() or any(ch.isdigit() for ch in name) for name in names)

    def test_unknown_modality_raises(self, mudataset: Dataset):
        with pytest.raises(KeyError, match="nonexistent"):
            mudataset.get_cell_line_features("nonexistent", mudataset.cell_line_ids[:1])


class TestDrugFeatures:
    """Test drug feature retrieval."""

    def test_chemberta(self, mudataset: Dataset):
        ids = mudataset.drug_ids[:4]
        features = mudataset.get_drug_features("chemberta", ids)
        assert features.shape == (4, CHEMBERTA_DIM)
        assert features.dtype == np.float32

    def test_morgan_fingerprint(self, mudataset: Dataset):
        features = mudataset.get_drug_features("morgan_fingerprint", mudataset.drug_ids)
        assert features.shape == (N_DRUGS, FINGERPRINT_BITS)

    def test_bpe_smiles(self, mudataset: Dataset):
        features = mudataset.get_drug_features("bpe_smiles", mudataset.drug_ids)
        assert features.shape == (N_DRUGS, BPE_LENGTH)

    def test_canonical_smiles_is_the_single_raw_drug_view(self, mudataset: Dataset):
        assert "canonical_smiles" in mudataset.response.var.columns
        assert mudataset.response.var["canonical_smiles"].notna().all()

    def test_missing_drug_raises(self, mudataset: Dataset):
        with pytest.raises(KeyError, match="nonexistent"):
            mudataset.get_drug_features("nonexistent", mudataset.drug_ids[:1])


class TestDrugGraphs:
    """Test drug graph access."""

    def test_get_drug_graphs(self, mudataset: Dataset):
        graphs = mudataset.get_drug_graphs(mudataset.drug_ids[:3])
        assert len(graphs) == 3
        for g in graphs:
            assert g is not None
            assert "x" in g
            assert "edge_index" in g
            assert "edge_attr" in g


class TestMetadata:
    """Test metadata access."""

    def test_cell_line_meta(self, mudataset: Dataset):
        meta = mudataset.cell_line_meta
        assert "cell_line_name" in meta.columns
        assert "tissue" in meta.columns

    def test_get_tissue(self, mudataset: Dataset):
        tissues = mudataset.get_tissue(mudataset.cell_line_ids[:5])
        assert len(tissues) == 5

    def test_get_tissue_unknown_id(self, mudataset: Dataset):
        tissues = mudataset.get_tissue(np.array(["FAKE_ID"]))
        assert len(tissues) == 1

    def test_enough_tissues_for_leave_tissue_out(self, mudataset: Dataset):
        tissues = np.unique(mudataset.get_tissue(mudataset.cell_line_ids))
        assert len(tissues) >= 3


class TestSubsetting:
    """Test subsetting operations."""

    def test_subset_cell_lines(self, mudataset: Dataset):
        ids = mudataset.cell_line_ids[:10]
        sub = mudataset.subset_cell_lines(ids)
        assert len(sub.cell_line_ids) == 10
        assert sub.response_matrix.shape == (10, N_DRUGS)

    def test_subset_drugs(self, mudataset: Dataset):
        ids = mudataset.drug_ids[:5]
        sub = mudataset.subset_drugs(ids)
        assert len(sub.drug_ids) == 5
        assert sub.response_matrix.shape == (N_CELL_LINES, 5)

    def test_subset_preserves_uns(self, mudataset: Dataset):
        sub = mudataset.subset_cell_lines(mudataset.cell_line_ids[:5])
        assert "drug_graphs" in sub.mdata.uns


class TestUns:
    """Test uns access."""

    def test_get_uns(self, mudataset: Dataset):
        bpe = mudataset.get_uns("bpe_codes")
        assert isinstance(bpe, str)

    def test_pathways_gmt_is_tab_delimited(self, mudataset: Dataset):
        gmt = mudataset.get_uns("pathways_gmt")
        assert all("\t" in line for line in gmt.strip().splitlines())

    def test_sparsego_carries_both_text_files(self, mudataset: Dataset):
        sparsego = mudataset.get_uns("sparsego")
        assert set(sparsego) == {"gene2ind", "ontology"}

    def test_get_uns_missing(self, mudataset: Dataset):
        with pytest.raises(KeyError, match="nonexistent"):
            mudataset.get_uns("nonexistent")


@pytest.fixture()
def records() -> list[dict]:
    """Collect the ``store`` calls a precompute run makes."""
    return []


class TestPrecompute:
    """Test explicit pre-computation of featurizer representations."""

    def test_default_config_is_prepended_to_explicit_configs(self, mudataset: Dataset, records: list[dict]):
        mudataset.precompute(stub_featurizer(records), [{"n_components": 2}])

        assert [call["hyperparameters"] for call in records] == [{}, {"n_components": 2}]

    def test_cell_line_side_precomputes_every_cell_line(self, mudataset: Dataset, records: list[dict]):
        mudataset.precompute(stub_featurizer(records), [])

        assert records[0]["n_entities"] == N_CELL_LINES

    def test_drug_side_precomputes_every_drug(self, mudataset: Dataset, records: list[dict]):
        mudataset.precompute(stub_featurizer(records, side="drug"), [])

        assert records[0]["n_entities"] == N_DRUGS

    def test_featurizer_without_compute_from_source_is_fitted_first(self, mudataset: Dataset, records: list[dict]):
        mudataset.precompute(stub_fit_transform_featurizer(records), [])

        assert records[0] == {"call": "fit"}
        assert records[1]["shape"] == (N_CELL_LINES, 3)

    def test_view_is_forwarded_to_the_featurizer_constructor(self, mudataset: Dataset, records: list[dict]):
        seen: list[dict] = []
        featurizer_cls = stub_featurizer(records)
        original_init = featurizer_cls.__init__

        def spy_init(self, **kwargs):
            seen.append(kwargs)
            original_init(self, **kwargs)

        featurizer_cls.__init__ = spy_init
        mudataset.precompute(featurizer_cls, [], view="methylation")

        assert seen == [{"view": "methylation"}]

    def test_int_hyperparameters_are_sampled(self, mudataset: Dataset, records: list[dict]):
        """An int asks for N sampled configs; the empty HP space short-circuits Optuna."""
        mudataset.precompute(stub_featurizer(records), 2)

        assert len(records) == 2


class TestPrecomputeSingle:
    """Test the eligibility branches ``precompute_all`` funnels every featurizer through."""

    def test_featurizer_not_marked_for_precomputation_is_skipped(self, mudataset: Dataset, records: list[dict]):
        mudataset._precompute_single(stub_featurizer(records, precompute=False), 1)

        assert records == []

    def test_entity_id_only_featurizer_is_skipped(self, mudataset: Dataset, records: list[dict]):
        mudataset._precompute_single(stub_featurizer(records, entity_id_only=True), 1)

        assert records == []

    def test_featurizer_with_missing_source_data_is_skipped(self, mudataset: Dataset, records: list[dict]):
        mudataset._precompute_single(stub_featurizer(records, source_views=("absent_modality",)), 1)

        assert records == []

    def test_view_featurizer_without_declared_input_views_is_skipped(self, mudataset: Dataset, records: list[dict]):
        featurizer_cls = stub_featurizer(records, requires_view=True, input_views=None)

        mudataset._precompute_single(featurizer_cls, 1)

        assert records == []

    def test_eligible_featurizer_is_precomputed(self, mudataset: Dataset, records: list[dict]):
        mudataset._precompute_single(stub_featurizer(records, source_views=("gene_expression",)), 1)

        assert len(records) == 1

    def test_featurizer_errors_are_swallowed(self, mudataset: Dataset, records: list[dict]):
        def raise_value_error(source, entity_ids):
            raise ValueError("no features")

        mudataset._precompute_single(stub_featurizer(records, compute=raise_value_error), 1)

        assert records == []

    def test_precompute_all_visits_every_registered_featurizer(self, mudataset: Dataset, monkeypatch):
        from drevalpy.registry.cell_line_featurizer import cell_line_featurizer_registry
        from drevalpy.registry.drug_featurizer import drug_featurizer_registry

        visited: list[str] = []
        monkeypatch.setattr(
            Dataset,
            "_precompute_single",
            lambda self, cls, n_variants: visited.append(cls.__name__),
        )

        mudataset.precompute_all(n_variants=1)

        expected = len(cell_line_featurizer_registry.list_names()) + len(drug_featurizer_registry.list_names())
        assert len(visited) == expected


class TestSourceAvailability:
    """Test the raw-source checks that gate precomputation and model applicability."""

    def test_canonical_smiles_is_found_on_the_response_var(self, mudataset: Dataset, records: list[dict]):
        drug_cls = stub_featurizer(records, side="drug")

        assert mudataset._has_source_data(("canonical_smiles",), drug_cls) is True

    def test_cell_line_modality_is_resolved_through_the_accessor_map(self, mudataset: Dataset, records: list[dict]):
        cell_line_cls = stub_featurizer(records)

        assert mudataset._has_source_data(("copy_number_variation_gistic",), cell_line_cls) is True

    def test_missing_cell_line_modality_is_reported_absent(self, mudataset: Dataset, records: list[dict]):
        assert mudataset._has_source_data(("absent_modality",), stub_featurizer(records)) is False

    def test_drug_view_is_looked_up_in_response_varm(self, mudataset: Dataset, records: list[dict]):
        drug_cls = stub_featurizer(records, side="drug")

        assert mudataset._has_source_data(("morgan_fingerprint",), drug_cls) is True

    def test_missing_drug_view_is_reported_absent(self, mudataset: Dataset, records: list[dict]):
        drug_cls = stub_featurizer(records, side="drug")

        assert mudataset._has_source_data(("absent_view",), drug_cls) is False

    @pytest.mark.parametrize(
        ("views", "available"),
        [
            pytest.param(("gene_expression",), True, id="modality"),
            pytest.param(("morgan_fingerprint",), True, id="varm-drug-view"),
            pytest.param(("pathway_features",), True, id="obsm-view"),
            pytest.param(("gene_expression", "chemberta"), True, id="both-sides"),
            pytest.param(("gene_expression", "absent"), False, id="one-missing"),
        ],
    )
    def test_required_views_span_modalities_varm_and_obsm(self, mudataset: Dataset, views, available):
        assert mudataset._has_required_views(views) is available
