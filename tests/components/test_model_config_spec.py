"""Tests for unified model spec resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from drevalpy.components.config import ModelConfig
from drevalpy.components.extensions import load_extensions
from drevalpy.components.model_config_spec import build_model_config_from_spec
from drevalpy.components.register_builtins import register_builtin_components


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_build_model_config_from_legacy_zoo_name() -> None:
    config = build_model_config_from_spec("ElasticNet")
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.type == "scaledGeneExpression"
    assert config.drug_featurizer is not None
    assert config.predictor.type == "elasticNet"


def test_build_model_config_from_legacy_name_with_hyperparameters() -> None:
    config = build_model_config_from_spec("ElasticNet", hyperparameters={"alpha": 0.2})
    assert config.predictor.hyperparameters["alpha"] == 0.2


def test_build_model_config_from_baseline_predictor_token() -> None:
    config = build_model_config_from_spec("naiveMean")
    assert config.predictor.type == "naiveMean"
    assert config.cell_line_featurizer is None
    assert config.drug_featurizer is None


def test_build_model_config_from_recipe_triple() -> None:
    config = build_model_config_from_spec("scaledGeneExpression:fingerprints:elasticNet")
    assert config.model_id == "scaledGeneExpression:fingerprints:elasticNet"


def test_build_model_config_from_literature_zoo_name() -> None:
    config = build_model_config_from_spec("DIPK")
    assert config.predictor.type == "dipk"
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.type == "multiViewStructured"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.type == "molgnet"


def test_model_config_from_spec_classmethod_matches_helper() -> None:
    helper_config = build_model_config_from_spec("RandomForest")
    class_config = ModelConfig.from_spec("RandomForest")
    assert helper_config.predictor.type == class_config.predictor.type
    assert helper_config.cell_line_featurizer is not None
    assert class_config.cell_line_featurizer is not None
    assert helper_config.cell_line_featurizer.type == class_config.cell_line_featurizer.type


def test_unknown_spec_raises_helpful_error() -> None:
    with pytest.raises(ValueError, match="Unknown model spec"):
        build_model_config_from_spec("definitelyNotARealModelName")


def test_external_extension_resolved_through_spec(tmp_path: Path) -> None:
    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    (ext_dir / "components.py").write_text(
        """
import numpy as np
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.predictors.baseline import BaselinePredictor
from drevalpy.components.registry import register_cell_line_featurizer, register_predictor

@register_cell_line_featurizer("resolverCellLine", description="ext", category="general_purpose")
class ResolverCellLineFeaturizer(CellLineFeaturizer):
    def fit(self, features, *, entity_ids=None):
        self._output_dim = 1
        return self
    def transform(self, features, entity_ids):
        return np.ones((len(entity_ids), 1), dtype=np.float32)
    @property
    def output_dim(self):
        return self._output_dim

@register_predictor("resolverPredictor", description="ext", category="general_purpose")
class ResolverPredictor(BaselinePredictor):
    def fit(self, x, y, *, pair_context=None):
        self._mean = float(np.mean(y))
    def predict(self, x, *, pair_context=None):
        return np.full(len(x), self._mean, dtype=np.float64)
""",
        encoding="utf-8",
    )
    zoo_file = tmp_path / "external_zoo.yaml"
    zoo_file.write_text(
        """
resolverEntry:
  cell_line_featurizer:
    type: resolverCellLine
  predictor:
    type: resolverPredictor
""",
        encoding="utf-8",
    )
    load_extensions(directories=[ext_dir], zoo_files=[zoo_file])
    config = ModelConfig.from_spec("resolverEntry")
    assert config.cell_line_featurizer is not None
    assert config.predictor.type == "resolverPredictor"
    config.validate()
