"""Tests for the public ``drevalpy.models.config`` package surface."""

from __future__ import annotations

from drevalpy.models import config


def test_primary_constructors_are_package_attributes() -> None:
    assert callable(config.from_spec)
    assert callable(config.from_yaml)
    assert callable(config.from_dict)
    assert callable(config.validate)


def test_from_spec_and_validate() -> None:
    cfg = config.from_spec("ElasticNet")
    assert cfg.predictor.name == "elasticNet"
    config.validate(cfg)


def test_long_legacy_names_are_not_exported() -> None:
    for name in (
        "model_config_from_spec",
        "model_config_from_yaml",
        "model_config_from_dict",
        "validate_model_config",
        "build_from_spec",
        "build_model_config_from_spec",
    ):
        assert not hasattr(config, name), name
