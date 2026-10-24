"""Tests for featurizer."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.models.config import FeaturizerConfig, SampleFeaturizerConfig

CONCAT = {"name": "concatFeaturizers", "featurizers": ["raw[gene_expression]", "raw[mutations]"]}


def test_view_validation() -> None:
    with pytest.raises(ValidationError, match="view must be a non-empty string when set"):
        SampleFeaturizerConfig(name="denseSample", view="   ")
    assert SampleFeaturizerConfig(name="denseSample", view="gene_expression").view == "gene_expression"
    with pytest.raises(ValidationError):
        FeaturizerConfig(name="landmarkGenes", views=["gene_expression"])  # type: ignore[call-arg]


@pytest.mark.parametrize("cls", [FeaturizerConfig, SampleFeaturizerConfig])
def test_one_key_shorthand_is_accepted_by_base_and_pinned(cls: type[FeaturizerConfig]) -> None:
    config = cls.model_validate({"pca[methylation]": {"n_components": 42}})
    assert (config.name, config.view, config.registry) == ("pca", "methylation", "sample")
    assert config.hyperparameter_space is not None
    assert config.hyperparameter_space["n_components"]["default"] == 42


def test_sequences_are_tuples_but_dump_as_json_lists() -> None:
    config = SampleFeaturizerConfig.model_validate(CONCAT)
    assert isinstance(config.featurizers, tuple)
    assert [child.view for child in config.featurizers] == ["gene_expression", "mutations"]
    dumped = config.model_dump(mode="json")
    assert isinstance(dumped["featurizers"], list)
