"""Building model configs: ``from_recipe`` / ``from_dict`` / ``from_yaml`` and recipe errors."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from drevalpy.models.config import from_dict, from_recipe, from_yaml, validate
from drevalpy.models.config.io import _reject_unknown_token
from drevalpy.models.spec import ModelSpec
from drevalpy.registry._extensions import load_extensions


def test_single_drug_recipes_infer_scope_without_a_drug_featurizer() -> None:
    config = from_recipe("scaledGeneExpression:singleDrugElasticNet")
    assert config.model_id == "scaledGeneExpression:singleDrugElasticNet"
    assert config.scope.value == "single_drug"
    assert config.drug_featurizer is None


def test_two_part_multi_drug_recipe_rejected() -> None:
    with pytest.raises(ValueError, match="Predictor 'elasticNet' requires a drug_featurizer"):
        from_recipe("scaledGeneExpression:elasticNet")


def test_recipe_with_plus_concat_and_bracket_views() -> None:
    config = from_recipe("view[expression]+view[mutations]:fingerprints+oneHot:randomForest")
    assert config.sample_featurizer is not None
    assert config.drug_featurizer is not None
    assert config.sample_featurizer.name == config.drug_featurizer.name == "concatFeaturizers"
    assert config.predictor.name == "randomForest"
    sample_children = config.sample_featurizer.featurizers
    drug_children = config.drug_featurizer.featurizers
    assert sample_children is not None and drug_children is not None
    assert [(child.name, child.view) for child in sample_children] == [("view", "expression"), ("view", "mutations")]
    assert [child.name for child in drug_children] == ["fingerprints", "oneHot"]
    assert config.model_id == "view[expression]+view[mutations]:fingerprints+oneHot:randomForest"


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("definitelyNotARealModelName", "Unknown model spec"),
        ("randomForest", "Predictor 'randomForest' requires a sample_featurizer"),
        ("scaledGeneExpression:fingerprints:elasticNet:extra", "Malformed model recipe"),
        ("scaledGeneExpression:fingerprints:bogusPredictor", "Unknown predictor 'bogusPredictor'"),
        ("bogusFeaturizer:fingerprints:elasticNet", r"^Unknown sample featurizer 'bogusFeaturizer'"),
    ],
)
def test_recipe_errors(spec: str, match: str) -> None:
    with pytest.raises(ValueError, match=match) as caught:
        from_recipe(spec)
    assert "Invalid model config" not in str(caught.value)
    assert "Value error, " not in str(caught.value)


def test_zoo_names_are_not_recipes() -> None:
    with pytest.raises(ValueError, match="Unknown model spec 'ElasticNet'"):
        from_recipe("ElasticNet")


def test_reject_unknown_token() -> None:
    _reject_unknown_token("randomForest")
    with pytest.raises(ValueError, match="Unknown model spec 'definitelyNotARealModelName'"):
        _reject_unknown_token("definitelyNotARealModelName")


def test_external_extension_resolved_through_spec(tmp_path: Path) -> None:
    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    (ext_dir / "components.py").write_text(
        """
import numpy as np
from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.registry.featurizer import register as register_featurizer

@register_featurizer("resolverSample", side="sample", description="ext")
class ResolverSampleFeaturizer(Featurizer):
    @classmethod
    def output_blocks(cls, config):
        return (BlockSpec("ext", FeatureFormat.NUMERIC_MATRIX),)

    def fit(self, source, train_ids, *, train_pair_ids=None, es_pair_ids=None):
        return self

    def transform(self, source, ids):
        values = np.ones((len(ids), 1), dtype=np.float32)
        return FeaturizerResult(ids, {"ext": numeric_feature_block(values)}, ("ext",))

@register_predictor(
    "resolverPredictor",
    description="ext",
)
class ResolverPredictor(FeatureFreePredictor):
    def _fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "response required"
            raise ValueError(msg)
        self._mean = float(np.mean(batch.response))

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.full(batch.n_pairs, self._mean, dtype=np.float64)
""",
        encoding="utf-8",
    )
    zoo_file = tmp_path / "external_zoo.yaml"
    zoo_file.write_text(
        """
resolverEntry:
  predictor: resolverPredictor
""",
        encoding="utf-8",
    )
    load_extensions(directories=[ext_dir], zoo_files=[zoo_file])
    config = ModelSpec.from_name("resolverEntry").config
    assert config.sample_featurizer is None
    assert config.predictor.name == "resolverPredictor"
    validate(config)


def test_from_recipe_forms() -> None:
    baseline = from_recipe("naiveMean")
    assert (baseline.predictor.name, baseline.sample_featurizer, baseline.drug_featurizer) == ("naiveMean", None, None)

    triple = from_recipe("scaledGeneExpression:fingerprints:elasticNet")
    assert triple.model_id == "scaledGeneExpression:fingerprints:elasticNet"


def test_from_dict_sections_and_recipe_strings() -> None:
    config = from_dict(
        {
            "sample_featurizer": "view[expression]+view[mutations]",
            "drug_featurizer": "fingerprints",
            "predictor": {"name": "randomForest", "defaults": {"n_estimators": 60}},
        }
    )
    assert config.sample_featurizer is not None
    assert config.sample_featurizer.name == "concatFeaturizers"
    assert config.predictor.hyperparameter_space is not None
    assert config.predictor.hyperparameter_space["n_estimators"]["default"] == 60
    assert from_dict({"predictor": "naiveMean"}).predictor.name == "naiveMean"


def test_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        yaml.safe_dump({"sample_featurizer": "constant", "drug_featurizer": "oneHot", "predictor": "naiveDrugMean"}),
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
        ({"predictor": "elasticNet"}, {}, r"Invalid model config: Predictor 'elasticNet' requires"),
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
