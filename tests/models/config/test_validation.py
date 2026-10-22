"""Tests for ModelConfig construction-time validation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.models.config import (
    DrugFeaturizerConfig,
    ModelConfig,
    PredictorConfig,
    SampleFeaturizerConfig,
    validate,
)
from drevalpy.types.data.batch.feature_block import BlockSpec
from tests.models.config._stubs import (
    register_block_predictor_stub,
    register_dense_trio,
    register_feature_free_predictor_stub,
    register_featurizer_stub,
    register_matrix_predictor_stub,
)
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


def _dense_model_config(**overrides) -> ModelConfig:
    """Build the valid dense triple, with named slots replaceable per test.

    :param overrides: Slot values replacing the dense defaults.
    :returns: A ``ModelConfig`` over the registered stubs.
    """
    slots = {
        "sample_featurizer": SampleFeaturizerConfig(name="denseSample", view="gene_expression"),
        "drug_featurizer": DrugFeaturizerConfig(name="denseDrug", view="fingerprints"),
        "predictor": PredictorConfig(name="densePred"),
    }
    slots.update(overrides)
    return ModelConfig(**slots)


def test_valid_dense_config_passes() -> None:
    register_dense_trio()

    validate(_dense_model_config())


def test_unknown_sample_featurizer_fails() -> None:
    register_dense_trio()
    with pytest.raises((ValueError, ValidationError), match="Unknown Sample featurizer"):
        _dense_model_config(sample_featurizer=SampleFeaturizerConfig(name="missing"))


def test_wrong_registry_is_coerced_by_slot_subclasses() -> None:
    register_dense_trio()
    config = ModelConfig.model_validate(
        {
            "sample_featurizer": {"name": "denseSample", "registry": "drug"},
            "drug_featurizer": {"name": "denseDrug", "registry": "sample"},
            "predictor": {"name": "densePred"},
        }
    )
    assert config.sample_featurizer is not None
    assert config.sample_featurizer.registry == "sample"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.registry == "drug"
    validate(config)


def test_graph_featurizer_with_matrix_predictor_fails() -> None:
    register_featurizer_stub("graphSample", side="sample", contract=FeatureFormat.GRAPH)
    register_featurizer_stub("denseDrug", side="drug")
    register_matrix_predictor_stub("densePred")

    with pytest.raises((ValueError, ValidationError), match="Sample featurizer contract|numeric_matrix"):
        _dense_model_config(sample_featurizer=SampleFeaturizerConfig(name="graphSample"))


def test_graph_format_match_passes_for_block_predictor() -> None:
    register_featurizer_stub("graphSample", side="sample", contract=FeatureFormat.GRAPH)
    register_featurizer_stub("graphDrug", side="drug", contract=FeatureFormat.GRAPH)
    register_block_predictor_stub(
        "graphPred",
        sample_contract=FeatureFormat.GRAPH,
        drug_contract=FeatureFormat.GRAPH,
    )

    config = ModelConfig(
        sample_featurizer=SampleFeaturizerConfig(name="graphSample"),
        drug_featurizer=DrugFeaturizerConfig(name="graphDrug"),
        predictor=PredictorConfig(name="graphPred"),
    )
    validate(config)


def test_block_schema_reports_missing_named_block() -> None:
    register_featurizer_stub(
        "cellBlocks",
        side="sample",
        output_block_specs=(BlockSpec("wrong_name", FeatureFormat.NUMERIC_MATRIX),),
    )
    register_featurizer_stub(
        "drugBlocks",
        side="drug",
        output_block_specs=(BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),),
    )
    register_block_predictor_stub(
        "blockPred",
        required_sample_block_specs=(BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),),
        required_drug_block_specs=(BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),),
    )

    with pytest.raises((ValueError, ValidationError), match="blockPred.*gene_expression.*numeric_matrix.*wrong_name"):
        ModelConfig(
            sample_featurizer=SampleFeaturizerConfig(name="cellBlocks"),
            drug_featurizer=DrugFeaturizerConfig(name="drugBlocks"),
            predictor=PredictorConfig(name="blockPred"),
        )


def test_builtin_featurizer_declares_output_block_specs() -> None:
    from drevalpy.components.featurizers.drug.fingerprints import FingerprintsFeaturizer

    assert FingerprintsFeaturizer.output_block_specs == (BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),)


def test_feature_free_predictor_without_featurizers_passes() -> None:
    register_feature_free_predictor_stub("naiveMean")

    config = ModelConfig(
        sample_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name="naiveMean"),
    )
    validate(config)


def test_feature_using_predictor_without_featurizers_fails() -> None:
    register_dense_trio()
    with pytest.raises((ValueError, ValidationError), match="requires featurizers"):
        _dense_model_config(sample_featurizer=None, drug_featurizer=None)


def test_baseline_tag_does_not_allow_missing_featurizers() -> None:
    from drevalpy.registry._builtins import register_builtin_components

    register_builtin_components()
    with pytest.raises((ValueError, ValidationError), match="requires featurizers"):
        ModelConfig(
            sample_featurizer=None,
            drug_featurizer=None,
            predictor=PredictorConfig(name="naiveMeanEffects"),
        )


def test_single_drug_scope_requires_identity_drug_featurizer() -> None:
    from drevalpy.registry._builtins import register_builtin_components

    register_builtin_components()
    with pytest.raises((ValueError, ValidationError), match="requires drug_featurizer='identity'"):
        ModelConfig(
            sample_featurizer=SampleFeaturizerConfig(name="scaledGeneExpression"),
            drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
            predictor=PredictorConfig(name="singleDrugElasticNet"),
        )


def test_single_drug_scope_accepts_identity_routing_featurizer() -> None:
    from drevalpy.registry._builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        sample_featurizer=SampleFeaturizerConfig(name="scaledGeneExpression"),
        drug_featurizer=DrugFeaturizerConfig(name="identity"),
        predictor=PredictorConfig(name="singleDrugElasticNet"),
    )
    validate(config)
