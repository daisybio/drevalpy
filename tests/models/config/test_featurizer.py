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


@pytest.mark.parametrize("cls", [FeaturizerConfig, CellLineFeaturizerConfig])
def test_one_key_shorthand_is_accepted_by_base_and_pinned(cls: type[FeaturizerConfig]) -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = cls.model_validate({"pca[methylation]": {"n_components": 42}})
    assert (config.name, config.view, config.registry) == ("pca", "methylation", "cell_line")
    assert config.hyperparameter_space is not None
    assert config.hyperparameter_space["n_components"]["default"] == 42


def test_list_sequence_fields_are_stored_as_tuples() -> None:
    """Pydantic coerces incoming lists to tuples, so no hand-written validator is needed."""
    config = FeaturizerConfig(name="landmarkGenes", views=["gene_expression", "mutations"])
    assert config.views == ("gene_expression", "mutations")


def test_bare_string_is_not_exploded_into_character_views() -> None:
    """A string must not be accepted as a sequence of single-character views."""
    with pytest.raises(ValidationError):
        FeaturizerConfig(name="landmarkGenes", views="gene_expression")  # type: ignore[arg-type]


def test_json_dump_renders_sequences_as_lists() -> None:
    """``mode="json"`` must yield plain lists so exported YAML stays hand-editable."""
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = CellLineFeaturizerConfig.model_validate(
        {"name": "concatFeaturizers", "featurizers": ["raw[gene_expression]", "raw[mutations]"]},
    )
    dumped = config.model_dump(mode="json")
    assert isinstance(dumped["featurizers"], list)
    assert [child["view"] for child in dumped["featurizers"]] == ["gene_expression", "mutations"]
