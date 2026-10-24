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
from drevalpy.registry._builtins import register_builtin_components
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
    slots = {
        "sample_featurizer": SampleFeaturizerConfig(name="denseSample", view="gene_expression"),
        "drug_featurizer": DrugFeaturizerConfig(name="denseDrug", view="fingerprints"),
        "predictor": PredictorConfig(name="densePred"),
    }
    slots.update(overrides)
    return ModelConfig(**slots)


def _numeric(name: str) -> tuple[BlockSpec, ...]:
    return (BlockSpec(name, FeatureFormat.NUMERIC_MATRIX),)


def test_valid_dense_config_passes_and_registry_is_coerced_by_slot_subclasses() -> None:
    register_dense_trio()
    validate(_dense_model_config())

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


def test_unknown_sample_featurizer_fails() -> None:
    register_dense_trio()
    with pytest.raises((ValueError, ValidationError), match="Unknown Sample featurizer"):
        _dense_model_config(sample_featurizer=SampleFeaturizerConfig(name="missing"))


def test_graph_featurizer_with_matrix_predictor_fails() -> None:
    register_featurizer_stub("graphSample", side="sample", contract=FeatureFormat.GRAPH)
    register_featurizer_stub("denseDrug", side="drug")
    register_matrix_predictor_stub("densePred")

    with pytest.raises((ValueError, ValidationError), match="Sample featurizer contract|numeric_matrix"):
        _dense_model_config(sample_featurizer=SampleFeaturizerConfig(name="graphSample"))


def test_graph_format_match_passes_for_block_predictor() -> None:
    register_featurizer_stub("graphSample", side="sample", contract=FeatureFormat.GRAPH)
    register_featurizer_stub("graphDrug", side="drug", contract=FeatureFormat.GRAPH)
    register_block_predictor_stub("graphPred", sample_contract=FeatureFormat.GRAPH, drug_contract=FeatureFormat.GRAPH)

    validate(
        ModelConfig(
            sample_featurizer=SampleFeaturizerConfig(name="graphSample"),
            drug_featurizer=DrugFeaturizerConfig(name="graphDrug"),
            predictor=PredictorConfig(name="graphPred"),
        )
    )


def test_block_schema_reports_missing_named_block() -> None:
    register_featurizer_stub("cellBlocks", side="sample", output_block_specs=_numeric("wrong_name"))
    register_featurizer_stub("drugBlocks", side="drug", output_block_specs=_numeric("fingerprints"))
    register_block_predictor_stub(
        "blockPred",
        required_sample_block_specs=_numeric("gene_expression"),
        required_drug_block_specs=_numeric("fingerprints"),
    )

    with pytest.raises((ValueError, ValidationError), match="blockPred.*gene_expression.*numeric_matrix.*wrong_name"):
        ModelConfig(
            sample_featurizer=SampleFeaturizerConfig(name="cellBlocks"),
            drug_featurizer=DrugFeaturizerConfig(name="drugBlocks"),
            predictor=PredictorConfig(name="blockPred"),
        )


def test_required_block_names_are_enforced_for_matrix_predictors() -> None:
    register_featurizer_stub("denseSample", side="sample")
    register_featurizer_stub("denseDrug", side="drug")
    register_matrix_predictor_stub("densePred").required_drug_blocks = ("smilesvec",)

    with pytest.raises(
        (ValueError, ValidationError),
        match=r"densePred.*needs drug block\(s\) \['smilesvec'\].*'denseDrug' emits \[fingerprints\]",
    ):
        _dense_model_config()


def test_required_block_names_accept_a_featurizer_emitting_them() -> None:
    register_featurizer_stub("cellBlocks", side="sample", output_block_specs=_numeric("gene_expression"))
    register_featurizer_stub("denseDrug", side="drug")
    register_matrix_predictor_stub("densePred").required_sample_blocks = ("gene_expression",)

    validate(_dense_model_config(sample_featurizer=SampleFeaturizerConfig(name="cellBlocks")))


def test_required_block_names_name_the_missing_block_and_what_is_emitted() -> None:
    register_featurizer_stub("cellBlocks", side="sample", output_block_specs=_numeric("wrong_name"))
    register_featurizer_stub("denseDrug", side="drug")
    register_block_predictor_stub("blockPred").required_sample_blocks = ("gene_expression", "tissue")

    with pytest.raises((ValueError, ValidationError), match=r"\['gene_expression', 'tissue'\].*emits \[wrong_name\]"):
        ModelConfig(
            sample_featurizer=SampleFeaturizerConfig(name="cellBlocks"),
            drug_featurizer=DrugFeaturizerConfig(name="denseDrug"),
            predictor=PredictorConfig(name="blockPred"),
        )


def test_builtin_featurizer_declares_output_block_specs() -> None:
    from drevalpy.components.featurizers.drug.fingerprints import FingerprintsFeaturizer

    assert FingerprintsFeaturizer.output_block_specs == _numeric("fingerprints")


def test_feature_free_predictor_without_featurizers_passes() -> None:
    register_feature_free_predictor_stub("naiveMean")

    validate(ModelConfig(sample_featurizer=None, drug_featurizer=None, predictor=PredictorConfig(name="naiveMean")))


def test_feature_using_predictors_need_featurizers() -> None:
    register_dense_trio()
    with pytest.raises((ValueError, ValidationError), match="requires featurizers"):
        _dense_model_config(sample_featurizer=None, drug_featurizer=None)


def test_baseline_tag_does_not_allow_missing_featurizers() -> None:
    register_builtin_components()
    with pytest.raises((ValueError, ValidationError), match="requires featurizers"):
        ModelConfig(sample_featurizer=None, drug_featurizer=None, predictor=PredictorConfig(name="naiveMeanEffects"))


def test_single_drug_scope_accepts_identity_routing_featurizer() -> None:
    register_builtin_components()
    validate(
        ModelConfig(
            sample_featurizer=SampleFeaturizerConfig(name="scaledGeneExpression"),
            drug_featurizer=DrugFeaturizerConfig(name="identity"),
            predictor=PredictorConfig(name="singleDrugElasticNet"),
        )
    )
