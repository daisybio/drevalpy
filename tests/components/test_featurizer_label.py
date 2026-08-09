"""Tests for qualified featurizer selectors and block labels."""

from __future__ import annotations

from drevalpy.components.core.fitting.featurizer_label import (
    featurizer_config_block_label,
    qualified_featurizer_selector,
    requires_explicit_view,
)


def test_qualified_selector_uses_view_brackets() -> None:
    assert qualified_featurizer_selector("pca", "gene_expression") == "pca[expression]"
    assert qualified_featurizer_selector("raw", "mutations") == "raw[mutations]"
    assert qualified_featurizer_selector("landmarkGenes") == "landmarkGenes"


def test_block_labels_match_qualified_selectors() -> None:
    assert featurizer_config_block_label("pca", "proteomics") == "pca[proteomics]"
    assert featurizer_config_block_label("fingerprints", None) == "fingerprints"


def test_requires_explicit_view() -> None:
    assert requires_explicit_view("raw")
    assert requires_explicit_view("pca")
    assert not requires_explicit_view("landmarkGenes")
