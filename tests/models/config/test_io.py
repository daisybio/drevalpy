"""Building model configs: ``from_spec`` / ``from_dict`` / ``from_yaml``, spec errors, and resolved values."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from drevalpy.models.config import ModelConfig, ResolvedModelConfig, from_dict, from_spec, from_yaml, validate
from drevalpy.models.factory import reject_unknown_spec, zoo_config
from drevalpy.models.tuning.search_space import resolve_model_config
from drevalpy.models.zoo import clear_external_zoo, zoo_model_config
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry._extensions import load_extensions
from drevalpy.types.enums.prediction_mode import PredictionMode
from tests.registry._helpers import restore_component_registries


@pytest.fixture(autouse=True)
def _register_components() -> Iterator[None]:
    # Evicting the extension matters: BUILTIN_*_NAMES are lazy singletons that would count it as a built-in.
    register_builtin_components()
    yield
    restore_component_registries()
    clear_external_zoo()


def test_zoo_names_resolve_to_their_components() -> None:
    config = from_spec("ElasticNet")
    assert isinstance(config, ModelConfig)
    assert config.sample_featurizer is not None
    assert config.sample_featurizer.name == "scaledGeneExpression"
    assert config.drug_featurizer is not None
    assert config.predictor.name == "elasticNet"

    dipk = from_spec("DIPK")
    assert isinstance(dipk, ModelConfig)
    assert dipk.predictor.name == "dipk"
    assert dipk.sample_featurizer is not None
    assert dipk.sample_featurizer.name == "concatFeaturizers"
    assert dipk.drug_featurizer is not None
    assert dipk.drug_featurizer.name == "molgnet"


def test_zoo_name_with_hyperparameters_is_resolved() -> None:
    config = from_spec("ElasticNet", hyperparameters={"alpha": 0.2})
    assert isinstance(config, ResolvedModelConfig)
    assert config.predictor_values()["alpha"] == 0.2


def test_zoo_name_prediction_mode_is_threaded_but_ignored_with_hyperparameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pins a long-standing quirk so a future fix is a deliberate change.
    from drevalpy.registry.predictor import get as get_predictor

    monkeypatch.setattr(get_predictor("elasticNet"), "supported_modes", frozenset(PredictionMode))

    template = from_spec("ElasticNet", prediction_mode=PredictionMode.CLASSIFICATION)
    assert not isinstance(template, ResolvedModelConfig)
    assert template.prediction_mode == PredictionMode.CLASSIFICATION

    resolved = from_spec("ElasticNet", hyperparameters={"alpha": 0.2}, prediction_mode=PredictionMode.CLASSIFICATION)
    assert isinstance(resolved, ResolvedModelConfig)
    assert resolved.template.prediction_mode == PredictionMode.REGRESSION


def test_single_drug_recipes_infer_scope_and_identity_routing() -> None:
    config = from_spec("scaledGeneExpression:identity:singleDrugElasticNet")
    two_part = from_spec("scaledGeneExpression:singleDrugElasticNet")
    assert isinstance(config, ModelConfig)
    assert isinstance(two_part, ModelConfig)
    assert config.model_id == two_part.model_id == "scaledGeneExpression:singleDrugElasticNet"
    assert config.scope.value == "single_drug"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "identity"
    assert two_part.drug_featurizer is not None
    assert two_part.drug_featurizer.name == "identity"


def test_two_part_multi_drug_recipe_rejected() -> None:
    with pytest.raises(ValueError, match="Predictor 'elasticNet' requires a drug_featurizer"):
        from_spec("scaledGeneExpression:elasticNet")


def test_recipe_with_plus_concat_and_bracket_views() -> None:
    config = from_spec("raw[expression]+raw[mutations]:fingerprints+identity:randomForest")
    assert isinstance(config, ModelConfig)
    assert config.sample_featurizer is not None
    assert config.drug_featurizer is not None
    assert config.sample_featurizer.name == config.drug_featurizer.name == "concatFeaturizers"
    assert config.predictor.name == "randomForest"
    sample_children = config.sample_featurizer.featurizers
    drug_children = config.drug_featurizer.featurizers
    assert sample_children is not None and drug_children is not None
    assert [(child.name, child.view) for child in sample_children] == [("raw", "expression"), ("raw", "mutations")]
    assert [child.name for child in drug_children] == ["fingerprints", "identity"]
    assert config.model_id == "concatFeaturizers:concatFeaturizers:randomForest"


def test_prediction_mode_accepts_a_string_or_the_enum() -> None:
    from_string = from_spec("ElasticNet", prediction_mode="regression")
    from_enum = from_spec("ElasticNet", prediction_mode=PredictionMode.REGRESSION)
    assert isinstance(from_string, ModelConfig)
    assert isinstance(from_enum, ModelConfig)
    assert from_string.prediction_mode == from_enum.prediction_mode == PredictionMode.REGRESSION
    with pytest.raises(ValueError, match="nonsense"):
        from_spec("ElasticNet", prediction_mode="nonsense")


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("definitelyNotARealModelName", "Unknown model spec"),
        ("randomForest", "Predictor 'randomForest' requires featurizers"),
        ("scaledGeneExpression:fingerprints:elasticNet:extra", "Malformed model recipe"),
        ("scaledGeneExpression:fingerprints:bogusPredictor", "Unknown Predictor: 'bogusPredictor'"),
        ("bogusFeaturizer:fingerprints:elasticNet", r"in recipe 'bogusFeaturizer:fingerprints:elasticNet'"),
    ],
)
def test_spec_errors(spec: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        from_spec(spec)


def test_reject_unknown_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    reject_unknown_spec("randomForest")
    with pytest.raises(ValueError, match="Unknown model spec 'definitelyNotARealModelName'"):
        reject_unknown_spec("definitelyNotARealModelName")

    monkeypatch.setattr("drevalpy.models.factory.list_predictors", lambda: [])
    reject_unknown_spec("dipk")
    monkeypatch.undo()

    monkeypatch.setattr("drevalpy.models.factory.is_known_builtin_predictor", lambda name: False)
    reject_unknown_spec("randomForest")


def test_zoo_config_returns_none_for_a_name_that_is_not_a_preset() -> None:
    assert zoo_config("definitelyNotARealModelName", None, PredictionMode.REGRESSION) is None
    assert zoo_config("ElasticNet", None, PredictionMode.REGRESSION) is not None


def test_external_extension_resolved_through_spec(tmp_path: Path) -> None:
    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    (ext_dir / "components.py").write_text(
        """
import numpy as np
from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.sample.base import SampleFeaturizer
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.registry.sample_featurizer import register as register_sample_featurizer

@register_sample_featurizer("resolverSample", description="ext", contract=FeatureFormat.NUMERIC_MATRIX)
class ResolverSampleFeaturizer(SampleFeaturizer):
    entity_id_only = True

    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        self._output_dim = 1
        return self
    def _transform_features(self, source, entity_ids):
        values = np.ones((len(entity_ids), 1), dtype=np.float32)
        return FeaturizerResult(values, {"ext": numeric_feature_block(values)})
    @property
    def output_dim(self):
        return self._output_dim

@register_predictor(
    "resolverPredictor",
    description="ext",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
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
    config = from_spec("resolverEntry")
    assert isinstance(config, ModelConfig)
    assert config.sample_featurizer is None
    assert config.predictor.name == "resolverPredictor"
    validate(config)


def test_from_spec_forms() -> None:
    baseline = from_spec("naiveMean")
    assert isinstance(baseline, ModelConfig)
    assert (baseline.predictor.name, baseline.sample_featurizer, baseline.drug_featurizer) == ("naiveMean", None, None)

    triple = from_spec("scaledGeneExpression:fingerprints:elasticNet")
    assert isinstance(triple, ModelConfig)
    assert triple.model_id == "scaledGeneExpression:fingerprints:elasticNet"


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


def test_resolve_separates_template_and_values() -> None:
    template = from_spec("ElasticNet")
    assert isinstance(template, ModelConfig)
    resolved = resolve_model_config(template)
    assert isinstance(resolved, ResolvedModelConfig)
    assert resolved.template.model_dump() == template.model_dump()
    assert resolved.predictor_values()["alpha"] == 1.0
    assert "predictor.elasticNet.alpha" in resolved.values


def test_explicit_hyperparameters_override_defaults() -> None:
    resolved = zoo_model_config("ElasticNet", {"alpha": 0.25})
    assert resolved.predictor_values()["alpha"] == 0.25
    assert resolved.values["predictor.elasticNet.alpha"] == 0.25


def test_featurizer_values_use_qualified_selectors() -> None:
    resolved = from_spec(
        "pca[expression]+pca[proteomics]:fingerprints:randomForest",
        hyperparameters={
            "sample_featurizer.pca[expression].n_components": 32,
            "sample_featurizer.pca[proteomics].n_components": 16,
        },
    )
    assert resolved.featurizer_values("sample", "pca[expression]")["n_components"] == 32
    assert resolved.featurizer_values("sample", "pca[proteomics]")["n_components"] == 16
