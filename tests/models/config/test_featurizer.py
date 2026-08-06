"""Tests for featurizer config schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.models.config import CellLineFeaturizerConfig, FeaturizerConfig


def test_empty_view_string_fails() -> None:
    with pytest.raises(ValidationError, match="view must be a non-empty string when set"):
        CellLineFeaturizerConfig(name="denseCellLine", view="   ")


def test_empty_views_list_fails() -> None:
    with pytest.raises(ValidationError, match="views must be a non-empty list when set"):
        FeaturizerConfig(name="landmarkGenes", views=[])


def test_blank_views_entry_fails() -> None:
    with pytest.raises(ValidationError, match="views must contain non-empty strings"):
        FeaturizerConfig(name="landmarkGenes", views=["gene_expression", "  "])


def test_non_empty_view_is_accepted() -> None:
    config = CellLineFeaturizerConfig(name="denseCellLine", view="gene_expression")
    assert config.view == "gene_expression"
