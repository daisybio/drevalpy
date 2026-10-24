"""Tests for the omics (sample) and drug halves of ``synthetic``."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.sample.gene_lists import view_gene_union
from drevalpy.data._synthetic_drugs import (
    BPE_LENGTH,
    CHEMBERTA_DIM,
    DRUG_VIEWS,
    FINGERPRINT_BITS,
    MAX_SYNTHETIC_DRUGS,
    SMILES,
    SMILESVEC_DIM,
    UNS_PAYLOADS,
    _bpe_codes,
    _graph_to_dict,
    _pathways_gmt,
    _sparsego_uns,
    add_drug_views,
    add_uns_payloads,
)
from drevalpy.data._synthetic_omics import (
    N_PATHWAYS,
    SAMPLE_VIEWS,
    add_sample_views,
    check_choices,
    cpg_ids,
    gene_symbols,
    omics_anndata,
    omics_coverage,
    payload_genes,
    view_feature_names,
)
from drevalpy.errors import InvalidInputError
from drevalpy.testing import build_synthetic_dataset

pytestmark = pytest.mark.slow


def test_check_choices_passes_known_names_and_reports_every_choice_for_an_unknown_one():
    check_choices("things", ["a", "b"], ("a", "b", "c"))

    with pytest.raises(ValueError, match=r"Unknown things \['z'\]; choose from \['a', 'b'\]"):
        check_choices("things", ["a", "z"], ("b", "a"))


class TestNames:
    def test_gene_symbols_are_the_packaged_landmark_prefix_and_too_many_ask_for_explicit_names(self):
        assert gene_symbols(3) == gene_symbols(5)[:3]
        assert len(set(gene_symbols(20))) == 20
        with pytest.raises(InvalidInputError, match="pass feature_names"):
            gene_symbols(100_000)

    def test_cpg_ids_are_distinct_genomic_windows(self):
        ids = cpg_ids(30)

        assert len(set(ids)) == 30
        assert ids[0] == "chr1:1000000-1000800"

    @pytest.mark.parametrize("view", ["gene_expression", "proteomics", "mutations", "methylation"])
    def test_default_names_are_the_shipped_genes_of_the_view(self, view):
        names = view_feature_names(view, None, 4)

        assert names == list(view_gene_union(view))
        assert len(set(names)) == len(names)
        assert all(name.startswith("chr") for name in names) == (view == "methylation")

    @pytest.mark.parametrize(
        ("view", "prefix"), [("proteomics", "SYNTHGENE"), ("methylation", "chr"), ("other", "SYNTHGENE")]
    )
    def test_a_wider_request_pads_after_the_shipped_genes(self, view, prefix):
        shipped = list(view_gene_union(view))

        names = view_feature_names(view, None, len(shipped) + 4)

        assert names[: len(shipped)] == shipped
        assert len(names) == len(set(names)) == len(shipped) + 4
        assert all(name.startswith(prefix) for name in names[len(shipped) :])

    def test_a_list_applies_to_every_view_and_a_mapping_only_to_the_views_it_names(self):
        mapping = {"methylation": ["m1"]}

        assert view_feature_names("methylation", ["a", "b"], 9) == ["a", "b"]
        assert view_feature_names("methylation", mapping, 9) == ["m1"]
        assert view_feature_names("proteomics", mapping, 3) == list(view_gene_union("proteomics"))


def test_omics_coverage_maps_none_sequences_and_mappings():
    assert omics_coverage(None, 10) == {}
    assert omics_coverage(["a", "b"], 10) == {"a": 10, "b": 10}
    assert omics_coverage({"a": 4.0}, 10) == {"a": 4}


class TestOmicsAnndata:
    SAMPLES = [f"S{index}" for index in range(30)]

    def _build(self, view: str, n_covered: int = 30, n_features: int = 6):
        rng = np.random.default_rng(0)
        factors = np.random.default_rng(1).normal(size=(30, 3))
        return omics_anndata(
            rng, view, self.SAMPLES, factors, n_covered=n_covered, feature_names=gene_symbols(n_features)
        )

    def test_it_covers_the_leading_samples_under_the_given_names_with_float32_expression_near_six(self):
        modality = self._build("gene_expression", n_covered=12)

        assert list(modality.obs_names) == self.SAMPLES[:12]
        assert list(modality.var_names) == gene_symbols(6)
        assert modality.X.dtype == np.float32
        assert self._build("gene_expression").X.mean() == pytest.approx(6.0, abs=1.5)

    def test_proteomics_is_floored_above_zero(self):
        rng = np.random.default_rng(0)
        factors = np.random.default_rng(1).normal(size=(500, 3)) * 4

        modality = omics_anndata(
            rng, "proteomics", [str(i) for i in range(500)], factors, n_covered=500, feature_names=["a", "b"]
        )

        assert modality.X.min() >= 0.5

    @pytest.mark.parametrize(
        ("view", "allowed"),
        [("mutations", {0.0, 1.0}), ("copy_number_variation_gistic", {-2.0, -1.0, 0.0, 1.0, 2.0})],
    )
    def test_discrete_views_only_take_their_levels(self, view, allowed):
        assert set(np.unique(self._build(view).X)) <= allowed

    def test_a_single_covered_sample_does_not_divide_by_zero(self):
        assert np.isfinite(self._build("methylation", n_covered=1).X).all()

    def test_a_view_carries_the_shared_factors(self):
        rng = np.random.default_rng(0)
        factors = np.random.default_rng(1).normal(size=(200, 3))
        samples = [str(i) for i in range(200)]
        first = omics_anndata(rng, "gene_expression", samples, factors, n_covered=200, feature_names=gene_symbols(40))

        rank = np.linalg.svd(first.X - first.X.mean(0), compute_uv=False)
        assert rank[3] < 0.5 * rank[2]


class TestPayloadGenes:
    def test_the_first_gene_axis_view_names_the_first_n_genes(self):
        data = build_synthetic_dataset(omics=["methylation", "proteomics"], n_features=5)
        built = {name: data.mdata.mod[name] for name in ("methylation", "proteomics")}

        assert payload_genes(built, 5) == list(view_gene_union("proteomics"))[:5]

    def test_without_a_gene_view_it_falls_back_to_landmark_symbols(self):
        assert payload_genes({}, 7) == gene_symbols(7)


class TestAddSampleViews:
    def test_pathway_features_fill_obsm(self):
        data = build_synthetic_dataset(n_samples=10)

        add_sample_views(data, SAMPLE_VIEWS, np.random.default_rng(0))

        assert data.mdata.mod["response"].obsm["pathway_features"].shape == (10, N_PATHWAYS)

    def test_an_unknown_view_is_rejected(self):
        with pytest.raises(ValueError, match="sample_views"):
            add_sample_views(build_synthetic_dataset(), ["nope"], np.random.default_rng(0))


GENES = [f"G{index}" for index in range(8)]


def test_every_smiles_parses_and_is_distinct_and_the_limit_is_the_molecule_count():
    from rdkit import Chem

    assert MAX_SYNTHETIC_DRUGS == len(SMILES)
    assert len(set(SMILES)) == len(SMILES)
    assert all(Chem.MolFromSmiles(value) is not None for value in SMILES)


class TestTextPayloads:
    @pytest.mark.parametrize(("genes", "n_per_set"), [(GENES, 5), (GENES[:3], 3)])
    def test_the_gmt_has_two_sets_of_up_to_five_genes(self, genes, n_per_set):
        rows = [line.split("\t") for line in _pathways_gmt(genes).splitlines()]

        assert [row[0] for row in rows] == ["SYNTH_PATHWAY_A", "SYNTH_PATHWAY_B"]
        assert all(len(row) - 2 >= n_per_set for row in rows)

    def test_sparsego_indexes_every_gene_and_links_it_to_a_term(self):
        payload = _sparsego_uns(GENES)

        assert payload["gene2ind"].splitlines()[2] == "2\tG2"
        gene_rows = [line for line in payload["ontology"].splitlines() if line.endswith("\tgene")]
        assert [row.split("\t")[1] for row in gene_rows] == GENES

    def test_bpe_codes_learn_merges_from_the_smiles(self):
        codes = _bpe_codes(SMILES[:8], num_symbols=10)

        assert codes.startswith("#version")
        assert len(codes.splitlines()) > 1


def test_graph_to_dict_stores_plain_typed_arrays():
    class Graph:
        x = [[1.0, 2.0]]
        edge_index = [[0], [0]]
        edge_attr = [[1.0]]

    stored = _graph_to_dict(Graph())

    assert {key: value.dtype for key, value in stored.items()} == {
        "x": np.float32,
        "edge_index": np.int64,
        "edge_attr": np.float32,
    }


class TestAddViewsAndPayloads:
    def test_drug_views_are_filled_at_their_width(self):
        data = build_synthetic_dataset(n_drugs=8)

        add_drug_views(data, DRUG_VIEWS, np.random.default_rng(0))

        varm = data.mdata.mod["response"].varm
        assert {name: varm[name].shape[1] for name in DRUG_VIEWS} == {
            "morgan_fingerprint": FINGERPRINT_BITS,
            "chemberta": CHEMBERTA_DIM,
            "bpe_smiles": BPE_LENGTH,
            "smilesvec": SMILESVEC_DIM,
        }

    def test_uns_payloads_are_all_filled(self):
        data = build_synthetic_dataset(n_drugs=8)

        add_uns_payloads(data, UNS_PAYLOADS, GENES)

        assert set(UNS_PAYLOADS) <= set(data.mdata.uns)
        assert len(data.mdata.uns["drug_graphs"]) == 8

    def test_unknown_views_and_payloads_are_rejected(self):
        with pytest.raises(ValueError, match="drug_views"):
            add_drug_views(build_synthetic_dataset(), ["nope"], np.random.default_rng(0))
        with pytest.raises(ValueError, match="uns_payloads"):
            add_uns_payloads(build_synthetic_dataset(), ["nope"], GENES)
