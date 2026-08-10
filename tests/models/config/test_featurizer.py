"""Tests for featurizer config schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.models.config import CellLineFeaturizerConfig, FeaturizerConfig


def test_empty_view_string_fails() -> None:
    with pytest.raises(ValidationError, match="view must be a non-empty string when set"):
        CellLineFeaturizerConfig(name="denseCellLine", view="   ")


def test_views_plural_is_rejected_as_unknown_field() -> None:
    """The vestigial ``views`` field is gone; ``extra="forbid"`` must reject it."""
    with pytest.raises(ValidationError):
        FeaturizerConfig(name="landmarkGenes", views=["gene_expression"])  # type: ignore[call-arg]


def test_non_empty_view_is_accepted() -> None:
    config = CellLineFeaturizerConfig(name="denseCellLine", view="gene_expression")
    assert config.view == "gene_expression"


@pytest.mark.parametrize("cls", [FeaturizerConfig, CellLineFeaturizerConfig])
def test_one_key_shorthand_is_accepted_by_base_and_pinned(cls: type[FeaturizerConfig]) -> None:
    from drevalpy.components.registry.register_builtins import register_builtin_components

    register_builtin_components()
    config = cls.model_validate({"pca[methylation]": {"n_components": 42}})
    assert (config.name, config.view, config.registry) == ("pca", "methylation", "cell_line")
    assert config.hyperparameter_space is not None
    assert config.hyperparameter_space["n_components"]["default"] == 42


def test_list_sequence_fields_are_stored_as_tuples() -> None:
    """Pydantic coerces incoming lists to tuples, so no hand-written validator is needed."""
    from drevalpy.components.registry.register_builtins import register_builtin_components

    register_builtin_components()
    config = CellLineFeaturizerConfig.model_validate(
        {"name": "concatFeaturizers", "featurizers": ["raw[gene_expression]", "raw[mutations]"]},
    )
    assert isinstance(config.featurizers, tuple)
    assert [child.view for child in config.featurizers] == ["gene_expression", "mutations"]


def test_json_dump_renders_sequences_as_lists() -> None:
    """``mode="json"`` must yield plain lists so exported YAML stays hand-editable."""
    from drevalpy.components.registry.register_builtins import register_builtin_components

    register_builtin_components()
    config = CellLineFeaturizerConfig.model_validate(
        {"name": "concatFeaturizers", "featurizers": ["raw[gene_expression]", "raw[mutations]"]},
    )
    dumped = config.model_dump(mode="json")
    assert isinstance(dumped["featurizers"], list)
    assert [child["view"] for child in dumped["featurizers"]] == ["gene_expression", "mutations"]
