"""Tests for drevalpy.models.config.spec.

Recipe and zoo resolution is what this module implements, so most cases drive it through
``from_spec``, which composes it. Tests naming ``recipe_payload`` or ``zoo_config`` pin the
individual steps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drevalpy.components.extensions import load_extensions
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models.config import from_spec, validate
from drevalpy.models.config.spec import recipe_payload, zoo_config
from drevalpy.types.model_scope import ModelScope
from drevalpy.types.prediction_mode import PredictionMode


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_build_model_config_from_zoo_name() -> None:
    config = from_spec("ElasticNet")
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "scaledGeneExpression"
    assert config.drug_featurizer is not None
    assert config.predictor.name == "elasticNet"


def test_build_model_config_from_zoo_name_with_hyperparameters() -> None:
    from drevalpy.models.config import ResolvedModelConfig

    config = from_spec("ElasticNet", hyperparameters={"alpha": 0.2})
    assert isinstance(config, ResolvedModelConfig)
    assert config.predictor_values()["alpha"] == 0.2


def test_zoo_name_prediction_mode_is_threaded_but_ignored_with_hyperparameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zoo presets honour ``prediction_mode`` only when no hyperparameters are given.

    The hyperparameters path returns the resolved config without applying the
    requested mode; this asserts that long-standing quirk so a future fix is a
    deliberate change rather than an accident.

    :param monkeypatch: Pytest fixture used to widen the predictor's supported modes.
    """
    from drevalpy.components.registry import get_predictor
    from drevalpy.models.config import ResolvedModelConfig

    monkeypatch.setattr(get_predictor("elasticNet"), "supported_modes", frozenset(PredictionMode))

    template = from_spec("ElasticNet", prediction_mode=PredictionMode.CLASSIFICATION)
    assert not isinstance(template, ResolvedModelConfig)
    assert template.prediction_mode == PredictionMode.CLASSIFICATION

    resolved = from_spec(
        "ElasticNet",
        hyperparameters={"alpha": 0.2},
        prediction_mode=PredictionMode.CLASSIFICATION,
    )
    assert isinstance(resolved, ResolvedModelConfig)
    assert resolved.template.prediction_mode == PredictionMode.REGRESSION


def test_build_model_config_from_baseline_predictor_token() -> None:
    config = from_spec("naiveMean")
    assert config.predictor.name == "naiveMean"
    assert config.cell_line_featurizer is None
    assert config.drug_featurizer is None


def test_build_model_config_from_recipe_triple() -> None:
    config = from_spec("scaledGeneExpression:fingerprints:elasticNet")
    assert config.model_id == "scaledGeneExpression:fingerprints:elasticNet"


def test_single_drug_recipe_infers_scope_and_identity_routing() -> None:
    config = from_spec("scaledGeneExpression:identity:singleDrugElasticNet")
    assert config.model_id == "scaledGeneExpression:singleDrugElasticNet"
    assert config.scope.value == "single_drug"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "identity"


def test_two_part_single_drug_recipe_matches_explicit_identity() -> None:
    two_part = from_spec("scaledGeneExpression:singleDrugElasticNet")
    three_part = from_spec("scaledGeneExpression:identity:singleDrugElasticNet")
    assert two_part.model_id == three_part.model_id == "scaledGeneExpression:singleDrugElasticNet"
    assert two_part.drug_featurizer is not None
    assert two_part.drug_featurizer.name == "identity"


def test_two_part_multi_drug_recipe_rejected() -> None:
    with pytest.raises(ValueError, match="two-part recipes require a single-drug predictor"):
        from_spec("scaledGeneExpression:elasticNet")


def test_build_model_config_from_recipe_triple_with_plus_concat() -> None:
    config = from_spec("raw[expression]+raw[mutations]:fingerprints+identity:randomForest")
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "concatFeaturizers"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "concatFeaturizers"
    assert config.predictor.name == "randomForest"
    cell_children = config.cell_line_featurizer.featurizers
    drug_children = config.drug_featurizer.featurizers
    assert cell_children is not None and drug_children is not None
    assert [child.name for child in cell_children] == ["raw", "raw"]
    assert cell_children[0].view == "gene_expression"
    assert cell_children[1].view == "mutations"
    assert [child.name for child in drug_children] == ["fingerprints", "identity"]
    assert config.model_id == "concatFeaturizers:concatFeaturizers:randomForest"


def test_build_model_config_from_recipe_triple_with_bracket_views() -> None:
    config = from_spec("raw[expression]+pca[proteomics]:identity:randomForest")
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "concatFeaturizers"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "identity"
    assert config.predictor.name == "randomForest"
    cell_children = config.cell_line_featurizer.featurizers
    assert cell_children is not None
    assert cell_children[0].name == "raw"
    assert cell_children[0].view == "gene_expression"
    assert cell_children[1].name == "pca"
    assert cell_children[1].view == "proteomics"


def test_build_model_config_from_literature_zoo_name() -> None:
    config = from_spec("DIPK")
    assert config.predictor.name == "dipk"
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "concatFeaturizers"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "molgnet"


def test_prediction_mode_accepts_a_string_or_the_enum() -> None:
    """The public entry point takes either, so callers need not import the enum."""
    from_string = from_spec("ElasticNet", prediction_mode="regression")
    from_enum = from_spec("ElasticNet", prediction_mode=PredictionMode.REGRESSION)
    assert from_string.prediction_mode == from_enum.prediction_mode == PredictionMode.REGRESSION


def test_invalid_prediction_mode_string_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonsense"):
        from_spec("ElasticNet", prediction_mode="nonsense")


def test_unknown_spec_raises_helpful_error() -> None:
    with pytest.raises(ValueError, match="Unknown model spec"):
        from_spec("definitelyNotARealModelName")


def test_recipe_payload_leaves_slots_as_recipe_strings() -> None:
    """The syntax step must not resolve names; that is ``from_dict``'s job."""
    payload = recipe_payload("raw[expression]+landmarkGenes:fingerprints:randomForest")
    assert payload["cell_line_featurizer"] == "raw[expression]+landmarkGenes"
    assert payload["drug_featurizer"] == "fingerprints"
    assert payload["predictor"] == "randomForest"


def test_two_part_recipe_payload_omits_the_drug_slot() -> None:
    """``ModelConfig`` injects the identity featurizer, so the payload leaves it unset."""
    payload = recipe_payload("scaledGeneExpression:singleDrugElasticNet")
    assert payload["drug_featurizer"] is None
    assert payload["scope"] is ModelScope.SINGLE_DRUG


def test_zoo_config_returns_none_for_a_name_that_is_not_a_preset() -> None:
    """Reporting a miss rather than raising is what lets ``from_spec`` fall through."""
    assert zoo_config("definitelyNotARealModelName", None, PredictionMode.REGRESSION) is None
    assert zoo_config("ElasticNet", None, PredictionMode.REGRESSION) is not None


def test_bare_predictor_requiring_featurizers_reports_the_missing_featurizers() -> None:
    """A registered predictor that needs featurizers is a config error, not an unknown spec."""
    with pytest.raises(ValueError, match="Predictor 'randomForest' requires featurizers"):
        from_spec("randomForest")


def test_malformed_recipe_keeps_the_grammar_error() -> None:
    """With a colon the intent is unambiguous, so the grammar's message survives."""
    with pytest.raises(ValueError, match="Malformed model recipe"):
        from_spec("scaledGeneExpression:fingerprints:elasticNet:extra")


def test_unknown_predictor_in_a_recipe_names_the_predictor() -> None:
    with pytest.raises(ValueError, match="Unknown Predictor: 'bogusPredictor'"):
        from_spec("scaledGeneExpression:fingerprints:bogusPredictor")


def test_recipe_validation_error_names_the_recipe() -> None:
    with pytest.raises(ValueError, match=r"in recipe 'bogusFeaturizer:fingerprints:elasticNet'"):
        from_spec("bogusFeaturizer:fingerprints:elasticNet")


def test_zoo_name_wins_over_a_bare_predictor_name() -> None:
    """``ElasticNet`` is a preset; the recipe path would reject it for missing featurizers."""
    config = from_spec("ElasticNet")
    assert config.cell_line_featurizer is not None


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
    config = from_spec("resolverEntry")
    assert config.cell_line_featurizer is None
    assert config.predictor.name == "resolverPredictor"
    validate(config)
