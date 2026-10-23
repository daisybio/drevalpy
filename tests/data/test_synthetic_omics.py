"""Tests for :mod:`drevalpy.data._synthetic_omics`, the sample-side half of ``synthetic``."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.sample.gene_lists import view_gene_union
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


class TestCheckChoices:
    def test_known_names_pass(self):
        check_choices("things", ["a", "b"], ("a", "b", "c"))

    def test_an_unknown_name_is_reported_with_every_choice(self):
        with pytest.raises(ValueError, match=r"Unknown things \['z'\]; choose from \['a', 'b'\]"):
            check_choices("things", ["a", "z"], ("b", "a"))


class TestNames:
    def test_gene_symbols_are_the_packaged_landmark_prefix(self):
        assert gene_symbols(3) == gene_symbols(5)[:3]
        assert len(set(gene_symbols(20))) == 20

    def test_too_many_symbols_ask_for_explicit_names(self):
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

    def test_one_list_applies_to_every_view(self):
        assert view_feature_names("methylation", ["a", "b"], 9) == ["a", "b"]

    def test_a_mapping_applies_only_to_the_views_it_names(self):
        mapping = {"methylation": ["m1"]}

        assert view_feature_names("methylation", mapping, 9) == ["m1"]
        assert view_feature_names("proteomics", mapping, 3) == list(view_gene_union("proteomics"))


class TestCoverage:
    def test_none_adds_no_view(self):
        assert omics_coverage(None, 10) == {}

    def test_a_sequence_gets_full_coverage(self):
        assert omics_coverage(["a", "b"], 10) == {"a": 10, "b": 10}

    def test_a_mapping_keeps_its_counts(self):
        assert omics_coverage({"a": 4.0}, 10) == {"a": 4}


class TestOmicsAnndata:
    SAMPLES = [f"S{index}" for index in range(30)]

    def _build(self, view: str, n_covered: int = 30, n_features: int = 6):
        rng = np.random.default_rng(0)
        factors = np.random.default_rng(1).normal(size=(30, 3))
        return omics_anndata(
            rng, view, self.SAMPLES, factors, n_covered=n_covered, feature_names=gene_symbols(n_features)
        )

    def test_it_covers_the_leading_samples_under_the_given_names(self):
        modality = self._build("gene_expression", n_covered=12)

        assert list(modality.obs_names) == self.SAMPLES[:12]
        assert list(modality.var_names) == gene_symbols(6)

    def test_expression_is_centred_on_six_and_float32(self):
        modality = self._build("gene_expression")

        assert modality.X.dtype == np.float32
        assert modality.X.mean() == pytest.approx(6.0, abs=1.5)

    def test_proteomics_is_floored_above_zero(self):
        rng = np.random.default_rng(0)
        factors = np.random.default_rng(1).normal(size=(500, 3)) * 4

        modality = omics_anndata(
            rng, "proteomics", [str(i) for i in range(500)], factors, n_covered=500, feature_names=["a", "b"]
        )

        assert modality.X.min() >= 0.5

    @pytest.mark.parametrize(
        ("view", "allowed"),
        [
            ("mutations", {0.0, 1.0}),
            ("copy_number_variation_gistic", {-2.0, -1.0, 0.0, 1.0, 2.0}),
        ],
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
        second = omics_anndata(rng, "mutations", samples, factors, n_covered=200, feature_names=gene_symbols(40))

        rank = np.linalg.svd(first.X - first.X.mean(0), compute_uv=False)
        assert rank[3] < 0.5 * rank[2]
        assert second.X.shape == first.X.shape


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
