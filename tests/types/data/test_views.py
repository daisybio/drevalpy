"""Public view names and the stored MuData modality names behind them."""

from __future__ import annotations

import pytest

from drevalpy.types.data.views import find_view, public_view_name, stored_view_name


class TestStoredViewName:
    @pytest.mark.parametrize(
        ("view", "expected"),
        [
            pytest.param("gene_expression", "gene_expression", id="identity"),
            pytest.param("copy_number_variation_gistic", "copy_number_variation", id="suffixed-name"),
            pytest.param("pathway_features", "pathway_features", id="non-omics-view"),
            pytest.param("drug_graph", "drug_graphs", id="drug-graphs-key"),
            pytest.param("", "", id="empty-string"),
        ],
    )
    def test_public_names_map_to_stored_names(self, view, expected):
        assert stored_view_name(view) == expected


class TestPublicViewName:
    @pytest.mark.parametrize(
        ("stored", "expected"),
        [
            pytest.param("copy_number_variation", "copy_number_variation_gistic", id="renamed"),
            pytest.param("methylation", "methylation", id="identity"),
            pytest.param("drug_graphs", "drug_graph", id="drug-graphs-key"),
            pytest.param("custom_matrix", "custom_matrix", id="unknown-name"),
        ],
    )
    def test_stored_names_map_back_to_public_names(self, stored, expected):
        assert public_view_name(stored) == expected

    @pytest.mark.parametrize(
        "view",
        [
            "gene_expression",
            "methylation",
            "mutations",
            "proteomics",
            "copy_number_variation_gistic",
            "drug_graph",
            "custom",
        ],
    )
    def test_round_trip(self, view):
        assert public_view_name(stored_view_name(view)) == view


class TestFindView:
    @pytest.mark.parametrize(
        ("available", "expected"),
        [
            pytest.param({"copy_number_variation"}, "copy_number_variation", id="stored-name"),
            pytest.param({"copy_number_variation_gistic"}, "copy_number_variation_gistic", id="public-name"),
            pytest.param(
                {"copy_number_variation", "copy_number_variation_gistic"},
                "copy_number_variation_gistic",
                id="public-name-wins",
            ),
            pytest.param({"gene_expression"}, None, id="absent"),
        ],
    )
    def test_either_stored_generation_is_found(self, available, expected):
        assert find_view("copy_number_variation_gistic", available) == expected

    @pytest.mark.parametrize(
        ("available", "expected"),
        [
            pytest.param({"drug_graphs", "sparsego"}, "drug_graphs", id="uns-key"),
            pytest.param({"drug_graph"}, "drug_graph", id="public-name"),
            pytest.param({"sparsego"}, None, id="absent"),
        ],
    )
    def test_the_drug_graph_view_is_found_under_its_uns_key(self, available, expected):
        assert find_view("drug_graph", available) == expected
