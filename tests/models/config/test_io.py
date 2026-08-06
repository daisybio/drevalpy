"""Tests for drevalpy.models.config.io."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models.config.io import (
    from_dict,
    from_spec,
    from_yaml,
)


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_model_config_from_predictor_only_spec() -> None:
    config = from_spec("naiveMean")
    assert config.predictor.name == "naiveMean"
    assert config.cell_line_featurizer is None
    assert config.drug_featurizer is None


def test_model_config_from_triple_spec() -> None:
    config = from_spec("scaledGeneExpression:fingerprints:elasticNet")
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "scaledGeneExpression"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "fingerprints"
    assert config.predictor.name == "elasticNet"


def test_model_config_from_triple_spec_with_plus_concat() -> None:
    config = from_spec("raw[expression]+raw[mutations]:fingerprints+identity:randomForest")
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "concatFeaturizers"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "concatFeaturizers"
    assert config.predictor.name == "randomForest"


def test_model_config_from_zoo_name() -> None:
    config = from_spec("ElasticNet")
    assert config.predictor.name == "elasticNet"


def test_from_dict_with_sections() -> None:
    config = from_dict(
        {
            "cell_line_featurizer": "scaledGeneExpression",
            "drug_featurizer": "fingerprints",
            "predictor": {"randomForest": {"n_estimators": 10}},
        }
    )
    assert config.predictor.hyperparameter_space is not None
    assert config.predictor.hyperparameter_space["n_estimators"]["default"] == 10


def test_from_dict_accepts_recipe_strings_in_slots() -> None:
    """Slots may hold recipe strings, which is what lets ``from_spec`` reuse this."""
    config = from_dict(
        {
            "cell_line_featurizer": "raw[expression]+raw[mutations]",
            "drug_featurizer": "fingerprints",
            "predictor": "randomForest",
        }
    )
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "concatFeaturizers"
    assert config.predictor.name == "randomForest"


def test_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "cell_line_featurizer": "constant",
                "drug_featurizer": "identity",
                "predictor": "naiveDrugMean",
            }
        ),
        encoding="utf-8",
    )
    config = from_yaml(path)
    assert config.predictor.name == "naiveDrugMean"


def test_from_dict_requires_predictor() -> None:
    with pytest.raises(ValueError, match="predictor"):
        from_dict({})


def test_from_dict_predictor_shorthand() -> None:
    config = from_dict({"predictor": "naiveMean"})
    assert config.predictor.name == "naiveMean"


def test_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown_key"):
        from_dict({"predictor": "naiveMean", "unknown_key": True})


def test_from_dict_rejects_invalid_prediction_mode() -> None:
    with pytest.raises(ValueError, match="prediction_mode"):
        from_dict({"predictor": "naiveMean", "prediction_mode": "invalid"})


def test_from_dict_source_label_is_included_in_the_error() -> None:
    with pytest.raises(ValueError, match=r"Invalid model config in my-label:"):
        from_dict({"predictor": "naiveMean", "unknown_key": True}, source="my-label")


def test_from_dict_error_without_source_omits_the_location() -> None:
    with pytest.raises(ValueError, match=r"Invalid model config: "):
        from_dict({"predictor": "naiveMean", "unknown_key": True})


def test_from_dict_field_level_error_names_the_field() -> None:
    with pytest.raises(ValueError, match=r"predictor: "):
        from_dict({"predictor": 123})


def test_from_dict_model_level_error_has_no_empty_field_prefix() -> None:
    """Whole-model errors carry an empty ``loc``, which must not render as a bare colon."""
    with pytest.raises(ValueError, match=r"Invalid model config: Value error, Predictor 'elasticNet' requires"):
        from_dict({"predictor": "elasticNet"})


def test_from_yaml_reports_path_on_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("predictor: naiveMean\nunknown_key: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(str(path))):
        from_yaml(path)


def test_from_yaml_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- naiveMean\n", encoding="utf-8")
    with pytest.raises(TypeError, match=re.escape(str(path))):
        from_yaml(path)
