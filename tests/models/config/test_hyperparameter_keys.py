"""Hyperparameter-key validation: the accepted keys of each component and ``validate_merged_mapping``."""

from __future__ import annotations

import pytest

from drevalpy.errors import UnknownNameError
from drevalpy.models.config import DrugFeaturizerConfig, ModelConfig, PredictorConfig, SampleFeaturizerConfig
from drevalpy.models.config._accepted_keys import featurizer_accepted_keys, leaf_selector, predictor_accepted_keys
from drevalpy.models.config._hp_key_validation import validate_merged_mapping


def _predictor(defaults: dict, space: dict, **extra):
    return type(
        "Predictor",
        (),
        {
            "get_default_hyperparameters": staticmethod(lambda: defaults),
            "get_hyperparameter_space": staticmethod(lambda: space),
            **extra,
        },
    )


@pytest.mark.parametrize(
    ("predictor", "expected"),
    [
        (_predictor({"alpha": 1.0}, {"l1_ratio": {"default": 0.5}}), {"alpha", "l1_ratio"}),
        (_predictor({"alpha": 1.0}, {}, non_tunable_hyperparameters={"max_iter": 1000}), {"alpha", "max_iter"}),
        (_predictor({}, {}), set()),
    ],
    ids=["defaults-and-space", "non-tunable", "empty"],
)
def test_predictor_accepted_keys(predictor, expected) -> None:
    assert predictor_accepted_keys(predictor) == expected


def test_featurizer_accepted_keys_read_the_registered_class_space() -> None:
    config = SampleFeaturizerConfig(name="pca", view="gene_expression")

    assert "n_components" in featurizer_accepted_keys(config, "sample")
    with pytest.raises(UnknownNameError):
        featurizer_accepted_keys(config, "tissue")  # type: ignore[arg-type]


def test_leaf_selector_qualifies_only_view_parametric_featurizers() -> None:
    assert leaf_selector(SampleFeaturizerConfig(name="pca", view="methylation")) == "pca[methylation]"
    assert leaf_selector(SampleFeaturizerConfig(name="identity")) == "identity"


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(
        sample_featurizer=SampleFeaturizerConfig(name="pca", view="gene_expression"),
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(name="elasticNet"),
    )


def _concat_config() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "sample_featurizer": ["scaledGeneExpression", {"pca[methylation]": {"n_components": 32}}],
            "drug_featurizer": "fingerprints",
            "predictor": "elasticNet",
        }
    )


class TestValidateMergedMapping:
    def test_accepts_declared_keys(self, config: ModelConfig) -> None:
        assert validate_merged_mapping(config, {}) is None
        merged = {
            "predictor.elasticNet.alpha": 0.5,
            "predictor.elasticNet.max_iter": 10,
            "sample_featurizer.pca[gene_expression].n_components": 32,
            "drug_featurizer.fingerprints.radius": 2,
            "drug_featurizer.fingerprints.n_bits": 1024,
        }
        assert validate_merged_mapping(config, merged) is None

    @pytest.mark.parametrize(
        "key",
        [
            "predictor.elasticNet.not_a_knob",
            "predictor.randomForest.n_estimators",
            "alpha",
            "sample_featurizer.pca.n_components",
            "drug_featurizer.fingerprints[smiles].radius",
        ],
        ids=["unknown-param", "another-predictors-param", "unqualified", "missing-view", "unexpected-view"],
    )
    def test_rejects_an_undeclared_key(self, config: ModelConfig, key: str) -> None:
        with pytest.raises(ValueError, match="Unknown hyperparameter"):
            validate_merged_mapping(config, {key: 1})

    def test_the_indexed_form_reports_the_migration_error(self, config: ModelConfig) -> None:
        with pytest.raises(ValueError, match="no longer supported"):
            validate_merged_mapping(config, {"sample_featurizer.pca.0.n_components": 32})

    def test_concat_children_are_addressable_but_not_the_parent(self) -> None:
        config = _concat_config()

        assert validate_merged_mapping(config, {"sample_featurizer.pca[methylation].n_components": 8}) is None
        with pytest.raises(ValueError, match="Unknown hyperparameter"):
            validate_merged_mapping(config, {"sample_featurizer.concatFeaturizers.n_components": 8})

    def test_stacks_without_declared_spaces_accept_no_featurizer_keys(self) -> None:
        no_knobs = ModelConfig(
            sample_featurizer=SampleFeaturizerConfig(name="scaledGeneExpression", view="gene_expression"),
            drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
            predictor=PredictorConfig(name="elasticNet"),
        )
        feature_free = ModelConfig(
            sample_featurizer=None, drug_featurizer=None, predictor=PredictorConfig(name="naiveMean")
        )

        with pytest.raises(ValueError, match="Unknown hyperparameter"):
            validate_merged_mapping(
                no_knobs, {"sample_featurizer.scaledGeneExpression[gene_expression].n_components": 8}
            )
        with pytest.raises(ValueError, match="Unknown hyperparameter"):
            validate_merged_mapping(feature_free, {"sample_featurizer.pca[gene_expression].n_components": 8})
