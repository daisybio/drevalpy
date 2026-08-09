"""Tests for omics view alias resolution."""

from __future__ import annotations

import pytest

from drevalpy.components.core.features.view_aliases import format_view_alias, resolve_omics_view


def test_resolve_expression_alias() -> None:
    assert resolve_omics_view("expression") == "gene_expression"


def test_resolve_cnv_alias() -> None:
    assert resolve_omics_view("cnv") == "copy_number_variation_gistic"


def test_format_view_alias_prefers_short_names() -> None:
    assert format_view_alias("gene_expression") == "expression"
    assert format_view_alias("copy_number_variation_gistic") == "cnv"


def test_resolve_rejects_unknown_view() -> None:
    with pytest.raises(ValueError, match="Unknown omics view"):
        resolve_omics_view("not_a_view")
