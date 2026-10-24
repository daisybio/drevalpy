"""Tests for drevalpy.models.config.io."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from drevalpy.models.config.io import from_dict, from_spec, from_yaml
from drevalpy.models.config.model import ModelConfig


def test_from_spec_forms() -> None:
    baseline = from_spec("naiveMean")
    assert isinstance(baseline, ModelConfig)
    assert (baseline.predictor.name, baseline.sample_featurizer, baseline.drug_featurizer) == ("naiveMean", None, None)

    triple = from_spec("scaledGeneExpression:fingerprints:elasticNet")
    assert isinstance(triple, ModelConfig)
    assert triple.sample_featurizer is not None
    assert triple.drug_featurizer is not None
    assert (triple.sample_featurizer.name, triple.drug_featurizer.name) == ("scaledGeneExpression", "fingerprints")

    concat = from_spec("raw[expression]+raw[mutations]:fingerprints+identity:randomForest")
    assert isinstance(concat, ModelConfig)
    assert concat.sample_featurizer is not None
    assert concat.drug_featurizer is not None
    assert concat.sample_featurizer.name == concat.drug_featurizer.name == "concatFeaturizers"

    zoo = from_spec("ElasticNet")
    assert isinstance(zoo, ModelConfig)
    assert zoo.predictor.name == "elasticNet"


def test_from_dict_sections_and_recipe_strings() -> None:
    config = from_dict(
        {
            "sample_featurizer": "raw[expression]+raw[mutations]",
            "drug_featurizer": "fingerprints",
            "predictor": {"randomForest": {"n_estimators": 10}},
        }
    )
    assert config.sample_featurizer is not None
    assert config.sample_featurizer.name == "concatFeaturizers"
    assert config.predictor.hyperparameter_space is not None
    assert config.predictor.hyperparameter_space["n_estimators"]["default"] == 10
    assert from_dict({"predictor": "naiveMean"}).predictor.name == "naiveMean"


def test_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        yaml.safe_dump({"sample_featurizer": "constant", "drug_featurizer": "identity", "predictor": "naiveDrugMean"}),
        encoding="utf-8",
    )

    assert from_yaml(path).predictor.name == "naiveDrugMean"


@pytest.mark.parametrize(
    ("payload", "kwargs", "match"),
    [
        ({}, {}, "predictor"),
        ({"predictor": "naiveMean", "unknown_key": True}, {}, "unknown_key"),
        ({"predictor": "naiveMean", "prediction_mode": "invalid"}, {}, "prediction_mode"),
        ({"predictor": "naiveMean", "unknown_key": True}, {"source": "my-label"}, r"Invalid model config in my-label:"),
        ({"predictor": "naiveMean", "unknown_key": True}, {}, r"Invalid model config: "),
        ({"predictor": 123}, {}, r"predictor: "),
        ({"predictor": "elasticNet"}, {}, r"Invalid model config: Value error, Predictor 'elasticNet' requires"),
    ],
    ids=["no-predictor", "unknown-key", "bad-mode", "source-label", "no-source", "field-level", "model-level"],
)
def test_from_dict_errors(payload, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        from_dict(payload, **kwargs)


def test_from_yaml_errors_report_the_path(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("predictor: naiveMean\nunknown_key: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(str(bad))):
        from_yaml(bad)

    listed = tmp_path / "list.yaml"
    listed.write_text("- naiveMean\n", encoding="utf-8")
    with pytest.raises(TypeError, match=re.escape(str(listed))):
        from_yaml(listed)
