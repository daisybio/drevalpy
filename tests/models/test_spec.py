"""Tests for the immutable public ModelSpec recipe."""

from __future__ import annotations

import re

import pytest

from drevalpy.errors import UnknownNameError
from drevalpy.models import ModelSpec
from drevalpy.models.config import from_spec
from drevalpy.models.spec import UnknownModelError


def test_from_name_returns_reusable_named_recipe() -> None:
    spec = ModelSpec.from_name("ElasticNet")

    assert spec.name == "ElasticNet"
    assert spec.config.predictor.name == "elasticNet"
    assert spec.resolve().predictor_values()["alpha"] == spec.default_hyperparameters["alpha"]
    assert all("." not in key for key in spec.default_hyperparameters)
    spec.default_hyperparameters["alpha"] = 999
    assert spec.default_hyperparameters["alpha"] != 999
    assert ModelSpec.from_name("NaivePredictor").hyperparameter_space == {}


def test_spec_snapshots_its_model_config() -> None:
    config = from_spec("ElasticNet")
    spec = ModelSpec("custom", config)

    assert spec.config is not config
    assert spec.config.model_dump(mode="python") == config.model_dump(mode="python")


def test_invalid_constructor_arguments_are_rejected() -> None:
    for name in ("", "  "):
        with pytest.raises(ValueError, match="non-empty"):
            ModelSpec(name, from_spec("ElasticNet"))
    with pytest.raises(TypeError, match="ModelConfig"):
        ModelSpec("bad", object())  # type: ignore[arg-type]


def test_from_name_reads_a_recipe_and_names_it_by_the_string_as_given() -> None:
    recipe = "pca[gene_expression]:fingerprints:elasticNet"

    spec = ModelSpec.from_name(recipe)

    assert spec.name == recipe
    assert spec.config.predictor.name == "elasticNet"
    assert spec.config.sample_featurizer.name == "pca"
    assert spec.config.sample_featurizer.view == "gene_expression"


def test_from_name_prefers_a_zoo_name_and_accepts_a_feature_free_token() -> None:
    assert ModelSpec.from_name("NaivePredictor").config.predictor.name == "naiveMean"
    assert ModelSpec.from_name("naiveMean").name == "naiveMean"


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("NotAModel", "Unknown model spec 'NotAModel'"),
        ("pca:elasticNet", "requires an explicit view"),
        ("raw[gene_expression]:fingerprints:naiveTissueMean", "needs sample block"),
        ("a:b:c:d", "Malformed model recipe"),
    ],
)
def test_from_name_error_names_both_readings_and_the_recipe_reason(name: str, reason: str) -> None:
    with pytest.raises(
        UnknownNameError, match=f"{re.escape(repr(name))} is not a zoo name and not a valid recipe: .*{reason}"
    ) as caught:
        ModelSpec.from_name(name)

    assert isinstance(caught.value, UnknownModelError)
    assert caught.value.name == name
    assert reason in caught.value.detail
    assert "Invalid model config" not in caught.value.detail
