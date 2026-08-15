"""Tests for merged hyperparameter key validation.

``validate_merged_mapping`` is the single public entry of
``drevalpy.models.config._hp_key_validation``; the accepted-key index it builds
is asserted through it rather than separately. The key grammar it delegates to
is covered in ``tests/models/test_hp_key_grammar.py``.
"""

from __future__ import annotations

import pytest

from drevalpy.models.config import (
    CellLineFeaturizerConfig,
    DrugFeaturizerConfig,
    ModelConfig,
    PredictorConfig,
)
from drevalpy.models.config._hp_key_validation import (
    _predictor_accepted_keys,
    validate_merged_mapping,
)
from drevalpy.registry._builtins import register_builtin_components


@pytest.fixture(autouse=True)
def _registry() -> None:
    """Register the built-ins the accepted-key index is derived from."""
    register_builtin_components()


@pytest.fixture
def config() -> ModelConfig:
    """A PCA / fingerprints / elastic-net stack, all three slots tunable."""
    return ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="pca", view="gene_expression"),
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
            pytest.param("cell_line_featurizer.pca[gene_expression].n_components", id="cell-line-featurizer-param"),
            pytest.param("drug_featurizer.fingerprints.radius", id="drug-featurizer-param"),
        ],
    )
    def test_accepts_a_declared_key(self, config: ModelConfig, key: str) -> None:
        assert validate_merged_mapping(config, {key: 1}) is None

    def test_accepts_several_keys_at_once(self, config: ModelConfig) -> None:
        merged = {
            "predictor.elasticNet.alpha": 0.5,
            "cell_line_featurizer.pca[gene_expression].n_components": 32,
            "drug_featurizer.fingerprints.n_bits": 1024,
        }

        assert validate_merged_mapping(config, merged) is None

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("predictor.elasticNet.not_a_knob", id="unknown-predictor-param"),
            pytest.param("predictor.randomForest.n_estimators", id="another-predictors-param"),
            pytest.param("alpha", id="unqualified-key"),
            pytest.param("cell_line_featurizer.pca.n_components", id="missing-view-qualifier"),
            pytest.param("drug_featurizer.fingerprints[smiles].radius", id="unexpected-view-qualifier"),
        ],
    )
    def test_rejects_an_undeclared_key(self, config: ModelConfig, key: str) -> None:
        with pytest.raises(ValueError, match="Unknown hyperparameter"):
            validate_merged_mapping(config, {key: 1})

    def test_the_indexed_form_reports_the_migration_error(self, config: ModelConfig) -> None:
        """The indexed check runs before the membership check, so the hint wins."""
        with pytest.raises(ValueError, match="no longer supported"):
            validate_merged_mapping(config, {"cell_line_featurizer.pca.0.n_components": 32})

    def test_expands_concat_children_into_leaf_selectors(self) -> None:
        config = ModelConfig.model_validate(
            {
                "cell_line_featurizer": ["scaledGeneExpression", {"pca[methylation]": {"n_components": 32}}],
                "drug_featurizer": "fingerprints",
                "predictor": "elasticNet",
            }
        )

        assert validate_merged_mapping(config, {"cell_line_featurizer.pca[methylation].n_components": 8}) is None

    def test_the_concat_parent_itself_is_not_addressable(self) -> None:
        config = ModelConfig.model_validate(
            {
                "cell_line_featurizer": ["scaledGeneExpression", {"pca[methylation]": {"n_components": 32}}],
                "drug_featurizer": "fingerprints",
                "predictor": "elasticNet",
            }
        )

        with pytest.raises(ValueError, match="Unknown hyperparameter"):
            validate_merged_mapping(config, {"cell_line_featurizer.concatFeaturizers.n_components": 8})

    def test_a_featurizer_without_a_declared_space_accepts_nothing(self, config: ModelConfig) -> None:
        no_knobs = ModelConfig(
            cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression", view="gene_expression"),
            drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
            predictor=PredictorConfig(name="elasticNet"),
        )

        with pytest.raises(ValueError, match="Unknown hyperparameter"):
            validate_merged_mapping(
                no_knobs, {"cell_line_featurizer.scaledGeneExpression[gene_expression].n_components": 8}
            )

    def test_a_feature_free_stack_accepts_no_featurizer_keys(self) -> None:
        config = ModelConfig(
            cell_line_featurizer=None,
            drug_featurizer=None,
            predictor=PredictorConfig(name="naiveMean"),
        )

        with pytest.raises(ValueError, match="Unknown hyperparameter"):
            validate_merged_mapping(config, {"cell_line_featurizer.pca[gene_expression].n_components": 8})


class TestPredictorAcceptedKeys:
    """``non_tunable_hyperparameters`` is accepted in either declared shape."""

    def test_merges_defaults_and_space(self) -> None:
        class Predictor:
            @staticmethod
            def get_default_hyperparameters() -> dict[str, object]:
                return {"alpha": 1.0}

            @staticmethod
            def get_hyperparameter_space() -> dict[str, object]:
                return {"l1_ratio": {"default": 0.5}}

        assert _predictor_accepted_keys(Predictor) == {"alpha", "l1_ratio"}

    @pytest.mark.parametrize(
        "non_tunable",
        [
            pytest.param({"max_iter": 1000}, id="mapping"),
            pytest.param(frozenset({"max_iter"}), id="frozenset"),
            pytest.param(["max_iter"], id="list"),
            pytest.param(("max_iter",), id="tuple"),
        ],
    )
    def test_includes_non_tunable_hyperparameters(self, non_tunable: object) -> None:
        class Predictor:
            non_tunable_hyperparameters = non_tunable

            @staticmethod
            def get_default_hyperparameters() -> dict[str, object]:
                return {"alpha": 1.0}

            @staticmethod
            def get_hyperparameter_space() -> dict[str, object]:
                return {}

        assert _predictor_accepted_keys(Predictor) == {"alpha", "max_iter"}

    def test_tolerates_an_absent_declaration(self) -> None:
        class Predictor:
            @staticmethod
            def get_default_hyperparameters() -> dict[str, object]:
                return {}

            @staticmethod
            def get_hyperparameter_space() -> dict[str, object]:
                return {}

        assert _predictor_accepted_keys(Predictor) == set()
