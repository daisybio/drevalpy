"""Tests for merged hyperparameter key validation.

``validate_merged_mapping`` is the single public entry of
``drevalpy.models.config._hp_key_validation``; the accepted-key index it builds
is asserted through it rather than separately. The per-component key sets are in
``test_accepted_keys.py`` and the key grammar in ``tests/models/test_hp_key_grammar.py``.
"""

from __future__ import annotations

import pytest

from drevalpy.models.config import (
    DrugFeaturizerConfig,
    ModelConfig,
    PredictorConfig,
    SampleFeaturizerConfig,
)
from drevalpy.models.config._hp_key_validation import validate_merged_mapping
from drevalpy.registry._builtins import register_builtin_components


@pytest.fixture(autouse=True)
def _registry() -> None:
    """Register the built-ins the accepted-key index is derived from."""
    register_builtin_components()


@pytest.fixture
def config() -> ModelConfig:
    """A PCA / fingerprints / elastic-net stack, all three slots tunable."""
    return ModelConfig(
        sample_featurizer=SampleFeaturizerConfig(name="pca", view="gene_expression"),
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(name="elasticNet"),
    )


class TestValidateMergedMapping:
    """Accepted keys are exactly those the configured components declare."""

    def test_an_empty_mapping_passes(self, config: ModelConfig) -> None:
        assert validate_merged_mapping(config, {}) is None

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("predictor.elasticNet.alpha", id="tunable-predictor-param"),
            pytest.param("predictor.elasticNet.max_iter", id="non-tunable-predictor-param"),
            pytest.param("sample_featurizer.pca[gene_expression].n_components", id="sample-featurizer-param"),
            pytest.param("drug_featurizer.fingerprints.radius", id="drug-featurizer-param"),
        ],
    )
    def test_accepts_a_declared_key(self, config: ModelConfig, key: str) -> None:
        assert validate_merged_mapping(config, {key: 1}) is None

    def test_accepts_several_keys_at_once(self, config: ModelConfig) -> None:
        merged = {
            "predictor.elasticNet.alpha": 0.5,
            "sample_featurizer.pca[gene_expression].n_components": 32,
            "drug_featurizer.fingerprints.n_bits": 1024,
        }

        assert validate_merged_mapping(config, merged) is None

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("predictor.elasticNet.not_a_knob", id="unknown-predictor-param"),
            pytest.param("predictor.randomForest.n_estimators", id="another-predictors-param"),
            pytest.param("alpha", id="unqualified-key"),
            pytest.param("sample_featurizer.pca.n_components", id="missing-view-qualifier"),
            pytest.param("drug_featurizer.fingerprints[smiles].radius", id="unexpected-view-qualifier"),
        ],
    )
    def test_rejects_an_undeclared_key(self, config: ModelConfig, key: str) -> None:
        with pytest.raises(ValueError, match="Unknown hyperparameter"):
            validate_merged_mapping(config, {key: 1})

    def test_the_indexed_form_reports_the_migration_error(self, config: ModelConfig) -> None:
        """The indexed check runs before the membership check, so the hint wins."""
        with pytest.raises(ValueError, match="no longer supported"):
            validate_merged_mapping(config, {"sample_featurizer.pca.0.n_components": 32})

    def test_expands_concat_children_into_leaf_selectors(self) -> None:
        config = ModelConfig.model_validate(
            {
                "sample_featurizer": ["scaledGeneExpression", {"pca[methylation]": {"n_components": 32}}],
                "drug_featurizer": "fingerprints",
                "predictor": "elasticNet",
            }
        )

        assert validate_merged_mapping(config, {"sample_featurizer.pca[methylation].n_components": 8}) is None

    def test_the_concat_parent_itself_is_not_addressable(self) -> None:
        config = ModelConfig.model_validate(
            {
                "sample_featurizer": ["scaledGeneExpression", {"pca[methylation]": {"n_components": 32}}],
                "drug_featurizer": "fingerprints",
                "predictor": "elasticNet",
            }
        )

        with pytest.raises(ValueError, match="Unknown hyperparameter"):
            validate_merged_mapping(config, {"sample_featurizer.concatFeaturizers.n_components": 8})

    def test_a_featurizer_without_a_declared_space_accepts_nothing(self, config: ModelConfig) -> None:
        no_knobs = ModelConfig(
            sample_featurizer=SampleFeaturizerConfig(name="scaledGeneExpression", view="gene_expression"),
            drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
            predictor=PredictorConfig(name="elasticNet"),
        )

        with pytest.raises(ValueError, match="Unknown hyperparameter"):
            validate_merged_mapping(
                no_knobs, {"sample_featurizer.scaledGeneExpression[gene_expression].n_components": 8}
            )

    def test_a_feature_free_stack_accepts_no_featurizer_keys(self) -> None:
        config = ModelConfig(
            sample_featurizer=None,
            drug_featurizer=None,
            predictor=PredictorConfig(name="naiveMean"),
        )

        with pytest.raises(ValueError, match="Unknown hyperparameter"):
            validate_merged_mapping(config, {"sample_featurizer.pca[gene_expression].n_components": 8})
