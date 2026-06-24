"""Tests for declarative config parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from drevalpy.components.config_io import (
    model_config_from_dict,
    model_config_from_spec,
    model_config_from_yaml,
)


def test_model_config_from_predictor_only_spec() -> None:
    config = model_config_from_spec("naiveMean")
    assert config.predictor.type == "naiveMean"
    assert config.cell_line_featurizer is None
    assert config.drug_featurizer is None


def test_model_config_from_triple_spec() -> None:
    config = model_config_from_spec("scaledGeneExpression:fingerprints:elasticNet")
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.type == "scaledGeneExpression"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.type == "fingerprints"
    assert config.predictor.type == "elasticNet"


def test_model_config_from_legacy_name() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = model_config_from_spec("ElasticNet")
    assert config.predictor.type == "elasticNet"


def test_model_config_from_dict_with_sections() -> None:
    config = model_config_from_dict(
        {
            "cell_line_featurizer": {"type": "scaledGeneExpression"},
            "drug_featurizer": {"type": "fingerprints", "registry": "drug"},
            "predictor": {"type": "randomForest", "hyperparameters": {"n_estimators": 10}},
        }
    )
    assert config.predictor.hyperparameters["n_estimators"] == 10


def test_model_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "predictor": {"type": "naiveDrugMean"},
            }
        ),
        encoding="utf-8",
    )
    config = model_config_from_yaml(path)
    assert config.predictor.type == "naiveDrugMean"


def test_model_config_from_dict_requires_predictor() -> None:
    with pytest.raises(ValueError, match="predictor"):
        model_config_from_dict({})


def test_model_config_from_dict_predictor_shorthand() -> None:
    config = model_config_from_dict({"predictor": "naiveMean"})
    assert config.predictor.type == "naiveMean"


def test_model_config_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown_key"):
        model_config_from_dict({"predictor": "naiveMean", "unknown_key": True})


def test_model_config_from_dict_rejects_invalid_prediction_mode() -> None:
    with pytest.raises(ValueError, match="prediction_mode"):
        model_config_from_dict({"predictor": "naiveMean", "prediction_mode": "invalid"})


def test_model_config_from_dict_rejects_invalid_predictor_shape() -> None:
    with pytest.raises(ValueError, match="predictor"):
        model_config_from_dict({"predictor": 123})


def test_model_config_from_yaml_reports_path_on_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("predictor: naiveMean\nunknown_key: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match=str(path)):
        model_config_from_yaml(path)
