"""Tests for ResolvedModelConfig template/value separation."""

from __future__ import annotations

from drevalpy.components.core.tuning.search_space import resolve_model_config
from drevalpy.components.registry.register_builtins import register_builtin_components
from drevalpy.models.config import ModelConfig, ResolvedModelConfig, from_spec
from drevalpy.models.zoo import zoo_model_config


def test_resolve_model_config_separates_template_and_values() -> None:
    register_builtin_components()
    template = from_spec("ElasticNet")
    assert isinstance(template, ModelConfig)
    resolved = resolve_model_config(template)
    assert isinstance(resolved, ResolvedModelConfig)
    assert resolved.template is template or resolved.template.model_dump() == template.model_dump()
    assert resolved.template.predictor.name == "elasticNet"
    assert resolved.predictor_values()["alpha"] == 1.0
    assert "predictor.elasticNet.alpha" in resolved.values


def test_explicit_hyperparameters_override_defaults() -> None:
    register_builtin_components()
    resolved = zoo_model_config("ElasticNet", {"alpha": 0.25})
    assert isinstance(resolved, ResolvedModelConfig)
    assert resolved.template.predictor.name == "elasticNet"
    assert resolved.predictor_values()["alpha"] == 0.25
    assert resolved.values["predictor.elasticNet.alpha"] == 0.25


def test_featurizer_values_use_qualified_selectors() -> None:
    register_builtin_components()
    resolved = from_spec(
        "pca[expression]+pca[proteomics]:fingerprints:randomForest",
        hyperparameters={
            "cell_line_featurizer.pca[expression].n_components": 32,
            "cell_line_featurizer.pca[proteomics].n_components": 16,
        },
    )
    assert isinstance(resolved, ResolvedModelConfig)
    assert resolved.featurizer_values("cell_line", "pca[expression]")["n_components"] == 32
    assert resolved.featurizer_values("cell_line", "pca[proteomics]")["n_components"] == 16
