"""Tests for the immutable public ModelSpec recipe."""

from __future__ import annotations

import pytest

from drevalpy.models import ModelSpec
from drevalpy.models.config import from_spec


def test_from_name_returns_reusable_named_recipe() -> None:
    spec = ModelSpec.from_name("ElasticNet")

    assert spec.name == "ElasticNet"
    assert spec.config.predictor.name == "elasticNet"
    assert spec.resolve().predictor_values()["alpha"] == spec.default_hyperparameters["alpha"]


def test_spec_snapshots_its_model_config() -> None:
    config = from_spec("ElasticNet")
    spec = ModelSpec("custom", config)

    assert spec.config is not config
    assert spec.config.model_dump(mode="python") == config.model_dump(mode="python")


@pytest.mark.parametrize("name", ["", "  "])
def test_empty_name_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ModelSpec(name, from_spec("ElasticNet"))


def test_non_model_config_is_rejected() -> None:
    with pytest.raises(TypeError, match="ModelConfig"):
        ModelSpec("bad", object())  # type: ignore[arg-type]


def test_public_hyperparameters_are_independent_mappings() -> None:
    spec = ModelSpec.from_name("ElasticNet")
    defaults = spec.default_hyperparameters
    defaults["alpha"] = 999

    assert spec.default_hyperparameters["alpha"] != 999


def test_default_spec_exposes_component_local_hyperparameters() -> None:
    spec = ModelSpec.from_name("ElasticNet")
    assert spec.default_hyperparameters["alpha"]
    assert all("." not in key for key in spec.default_hyperparameters)


def test_feature_free_spec_has_no_tunable_space() -> None:
    spec = ModelSpec.from_name("NaivePredictor")
    assert spec.hyperparameter_space == {}
