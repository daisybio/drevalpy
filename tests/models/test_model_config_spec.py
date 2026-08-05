"""Tests for drevalpy.models.model_config_spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from drevalpy.components.extensions import load_extensions
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models.config import ModelConfig
from drevalpy.models.model_config_spec import build_model_config_from_spec


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_build_model_config_from_zoo_name() -> None:
    config = build_model_config_from_spec("ElasticNet")
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "scaledGeneExpression"
    assert config.drug_featurizer is not None
    assert config.predictor.name == "elasticNet"


def test_build_model_config_from_zoo_name_with_hyperparameters() -> None:
    config = build_model_config_from_spec("ElasticNet", hyperparameters={"alpha": 0.2})
    assert config.predictor.hyperparameters["alpha"] == 0.2


def test_build_model_config_from_baseline_predictor_token() -> None:
    config = build_model_config_from_spec("naiveMean")
    assert config.predictor.name == "naiveMean"
    assert config.cell_line_featurizer is None
    assert config.drug_featurizer is None


def test_build_model_config_from_recipe_triple() -> None:
    config = build_model_config_from_spec("scaledGeneExpression:fingerprints:elasticNet")
    assert config.model_id == "scaledGeneExpression:fingerprints:elasticNet"


def test_single_drug_recipe_infers_scope_and_identity_routing() -> None:
    config = build_model_config_from_spec("scaledGeneExpression:identity:singleDrugElasticNet")
    assert config.model_id == "scaledGeneExpression:singleDrugElasticNet"
    assert config.scope.value == "single_drug"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "identity"


def test_two_part_single_drug_recipe_matches_explicit_identity() -> None:
    two_part = build_model_config_from_spec("scaledGeneExpression:singleDrugElasticNet")
    three_part = build_model_config_from_spec("scaledGeneExpression:identity:singleDrugElasticNet")
    assert two_part.model_id == three_part.model_id == "scaledGeneExpression:singleDrugElasticNet"
    assert two_part.drug_featurizer is not None
    assert two_part.drug_featurizer.name == "identity"


def test_two_part_multi_drug_recipe_rejected() -> None:
    with pytest.raises(ValueError, match="two-part recipes require a single-drug predictor"):
        build_model_config_from_spec("scaledGeneExpression:elasticNet")


def test_build_model_config_from_recipe_triple_with_plus_concat() -> None:
    config = build_model_config_from_spec("raw[expression]+raw[mutations]:fingerprints+identity:randomForest")
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "concatFeaturizers"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "concatFeaturizers"
    assert config.predictor.name == "randomForest"
    cell_children = config.cell_line_featurizer.hyperparameters["featurizers"]
    drug_children = config.drug_featurizer.hyperparameters["featurizers"]
    assert [child["name"] for child in cell_children] == ["raw", "raw"]
    assert cell_children[0]["view"] == "gene_expression"
    assert cell_children[1]["view"] == "mutations"
    assert [child["name"] for child in drug_children] == ["fingerprints", "identity"]
    assert config.model_id == "concatFeaturizers:concatFeaturizers:randomForest"


def test_build_model_config_from_recipe_triple_with_bracket_views() -> None:
    config = build_model_config_from_spec("raw[expression]+pca[proteomics]:identity:randomForest")
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "concatFeaturizers"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "identity"
    assert config.predictor.name == "randomForest"
    cell_children = config.cell_line_featurizer.hyperparameters["featurizers"]
    assert cell_children[0]["name"] == "raw"
    assert cell_children[0]["view"] == "gene_expression"
    assert cell_children[1]["name"] == "pca"
    assert cell_children[1]["view"] == "proteomics"


def test_build_model_config_from_literature_zoo_name() -> None:
    config = build_model_config_from_spec("DIPK")
    assert config.predictor.name == "dipk"
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "concatFeaturizers"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "molgnet"


def test_model_config_from_spec_classmethod_matches_helper() -> None:
    helper_config = build_model_config_from_spec("RandomForest")
    class_config = ModelConfig.from_spec("RandomForest")
    assert helper_config.predictor.name == class_config.predictor.name
    assert helper_config.cell_line_featurizer is not None
    assert class_config.cell_line_featurizer is not None
    assert helper_config.cell_line_featurizer.name == class_config.cell_line_featurizer.name


def test_unknown_spec_raises_helpful_error() -> None:
    with pytest.raises(ValueError, match="Unknown model spec"):
        build_model_config_from_spec("definitelyNotARealModelName")


def test_external_extension_resolved_through_spec(tmp_path: Path) -> None:
    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    (ext_dir / "components.py").write_text(
        """
import numpy as np
from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.components.registry import register_cell_line_featurizer, register_predictor

@register_cell_line_featurizer(
    "resolverCellLine",
    description="ext",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ResolverCellLineFeaturizer(CellLineFeaturizer):
    def fit(self, features, *, entity_ids=None):
        self._output_dim = 1
        return self
    def transform(self, features, entity_ids):
        return np.ones((len(entity_ids), 1), dtype=np.float32)
    @property
    def output_dim(self):
        return self._output_dim

@register_predictor(
    "resolverPredictor",
    description="ext",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class ResolverPredictor(FeatureFreePredictor):
    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "response required"
            raise ValueError(msg)
        self._mean = float(np.mean(batch.response))

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
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
    config = ModelConfig.from_spec("resolverEntry")
    assert config.cell_line_featurizer is None
    assert config.predictor.name == "resolverPredictor"
    config.validate()
