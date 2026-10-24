"""Tests for the portable file-name helper."""

from __future__ import annotations

import pytest

from drevalpy.utils.filenames import safe_filename


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ElasticNet", "ElasticNet"),
        ("pca[gene_expression]:fingerprints:ridge", "pca[gene_expression]_fingerprints_ridge"),
        ('a<b>c"d/e\\f|g?h*i', "a_b_c_d_e_f_g_h_i"),
        ("tab\there", "tab_here"),
        ("trailing. ", "trailing"),
        ("...", "_"),
    ],
)
def test_safe_filename(name: str, expected: str) -> None:
    assert safe_filename(name) == expected
