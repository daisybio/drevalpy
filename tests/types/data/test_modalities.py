"""Tests for the omics-name to modality-key resolution layer.

Every omics access in the package funnels through these three functions, so the
tests pin the property that makes the double naming safe: the name as written
always wins over the accessor map, which is what lets one code path read both
dataset generations.
"""

from __future__ import annotations

import pytest

from drevalpy.types.data.modalities import (
    OMICS_ACCESSORS,
    backing_modality,
    public_omics_name,
    resolve_omics_accessor,
)


class TestAccessorMap:
    def test_map_is_read_only(self):
        with pytest.raises(TypeError):
            OMICS_ACCESSORS["gene_expression"] = "something_else"  # type: ignore[index]

    def test_map_is_injective(self):
        assert len(set(OMICS_ACCESSORS.values())) == len(OMICS_ACCESSORS)

    def test_copy_number_is_the_only_non_identity_entry(self):
        non_identity = {public for public, accessor in OMICS_ACCESSORS.items() if public != accessor}

        assert non_identity == {"copy_number_variation_gistic"}


class TestResolveOmicsAccessor:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            pytest.param("gene_expression", "gene_expression", id="identity-entry"),
            pytest.param("copy_number_variation_gistic", "copy_number_variation", id="suffixed-name"),
            pytest.param("pathway_features", "pathway_features", id="non-omics-view"),
            pytest.param("", "", id="empty-string"),
        ],
    )
    def test_public_names_resolve_to_stored_keys(self, name, expected):
        assert resolve_omics_accessor(name) == expected


class TestPublicOmicsName:
    @pytest.mark.parametrize(
        ("accessor", "expected"),
        [
            pytest.param("copy_number_variation", "copy_number_variation_gistic", id="stored-key"),
            pytest.param("methylation", "methylation", id="identity-entry"),
            pytest.param("custom_matrix", "custom_matrix", id="non-omics-view"),
        ],
    )
    def test_stored_keys_resolve_back_to_public_names(self, accessor, expected):
        assert public_omics_name(accessor) == expected

    def test_round_trip_is_lossless_for_every_registered_view(self):
        assert all(public_omics_name(resolve_omics_accessor(name)) == name for name in OMICS_ACCESSORS)


class TestBackingModality:
    def test_the_name_as_written_wins_over_the_accessor_map(self):
        """A file already carrying the suffixed name is read directly, with no rename."""
        available = {"copy_number_variation_gistic", "copy_number_variation"}

        assert backing_modality("copy_number_variation_gistic", available) == "copy_number_variation_gistic"

    def test_the_accessor_map_is_the_fallback(self):
        assert backing_modality("copy_number_variation_gistic", {"copy_number_variation"}) == "copy_number_variation"

    def test_a_view_absent_from_the_file_is_unbacked(self):
        assert backing_modality("proteomics", {"gene_expression"}) is None

    def test_a_non_omics_view_is_matched_by_name_only(self):
        assert backing_modality("pathway_features", {"pathway_features"}) == "pathway_features"

    def test_an_unbacked_non_omics_view_is_none(self):
        assert backing_modality("pathway_features", set()) is None

    def test_any_container_of_keys_is_accepted(self):
        """Callers pass ``mdata.mod``, sets and lists interchangeably."""
        assert backing_modality("methylation", ["methylation"]) == "methylation"
