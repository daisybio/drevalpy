"""ModelConfig construction-time validation against the registries, and output block-spec resolution."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._featurizer_label import CONCAT
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictorConfig, validate
from drevalpy.registry.featurizer import featurizer_registry
from drevalpy.types.data.batch.feature_block import BlockSpec
from tests.models.config._stubs import (
    register_block_predictor_stub,
    register_dense_trio,
    register_feature_free_predictor_stub,
    register_featurizer_stub,
    register_matrix_predictor_stub,
)
from tests.registry._helpers import empty_component_registries, restore_builtin_components


@pytest.fixture(autouse=True)
def _clear_registries() -> None:
    empty_component_registries()


def _dense_model_config(**overrides) -> ModelConfig:
    slots = {
        "sample_featurizer": FeaturizerConfig(name="denseSample", side="sample", view="gene_expression"),
        "drug_featurizer": FeaturizerConfig(name="denseDrug", side="drug", view="fingerprints"),
        "predictor": PredictorConfig(name="densePred"),
    }
    slots.update(overrides)
    return ModelConfig(**slots)


def _numeric(name: str) -> tuple[BlockSpec, ...]:
    return (BlockSpec(name, FeatureFormat.NUMERIC_MATRIX),)


def test_valid_dense_config_passes() -> None:
    register_dense_trio()
    validate(_dense_model_config())

    config = ModelConfig.model_validate(
        {
            "sample_featurizer": {"name": "denseSample"},
            "drug_featurizer": {"name": "denseDrug"},
            "predictor": {"name": "densePred"},
        }
    )
    validate(config)


def test_featurizer_in_a_slot_of_a_side_it_does_not_support_fails() -> None:
    register_dense_trio()
    with pytest.raises(
        (ValueError, ValidationError),
        match="'denseDrug' is a drug featurizer and cannot be used on the sample side",
    ):
        _dense_model_config(sample_featurizer=FeaturizerConfig(name="denseDrug", side="sample"))


def test_unknown_sample_featurizer_fails() -> None:
    register_dense_trio()
    with pytest.raises((ValueError, ValidationError), match="Unknown sample featurizer 'missing'"):
        _dense_model_config(sample_featurizer=FeaturizerConfig(name="missing", side="sample"))


def test_graph_featurizer_with_matrix_predictor_fails() -> None:
    register_featurizer_stub("graphSample", side="sample", output_format=FeatureFormat.GRAPH)
    register_featurizer_stub("denseDrug", side="drug")
    register_matrix_predictor_stub("densePred")

    with pytest.raises((ValueError, ValidationError), match="Sample featurizer output_format|numeric_matrix"):
        _dense_model_config(sample_featurizer=FeaturizerConfig(name="graphSample", side="sample"))


def test_graph_format_match_passes_for_block_predictor() -> None:
    register_featurizer_stub("graphSample", side="sample", output_format=FeatureFormat.GRAPH)
    register_featurizer_stub("graphDrug", side="drug", output_format=FeatureFormat.GRAPH)
    register_block_predictor_stub("graphPred")

    validate(
        ModelConfig(
            sample_featurizer=FeaturizerConfig(name="graphSample", side="sample"),
            drug_featurizer=FeaturizerConfig(name="graphDrug", side="drug"),
            predictor=PredictorConfig(name="graphPred"),
        )
    )


def test_block_schema_reports_missing_named_block() -> None:
    register_featurizer_stub("cellBlocks", side="sample", output_block_specs=_numeric("wrong_name"))
    register_featurizer_stub("drugBlocks", side="drug", output_block_specs=_numeric("fingerprints"))
    register_block_predictor_stub(
        "blockPred",
        sample_blocks=_numeric("gene_expression"),
        drug_blocks=_numeric("fingerprints"),
    )

    with pytest.raises(
        (ValueError, ValidationError),
        match=r"blockPred.*needs sample block\(s\) \['gene_expression'\].*emits \[wrong_name\]",
    ):
        ModelConfig(
            sample_featurizer=FeaturizerConfig(name="cellBlocks", side="sample"),
            drug_featurizer=FeaturizerConfig(name="drugBlocks", side="drug"),
            predictor=PredictorConfig(name="blockPred"),
        )


def test_matrix_inputs_accept_any_numeric_blocks() -> None:
    register_featurizer_stub("cellBlocks", side="sample", output_block_specs=_numeric("wrong_name"))
    register_featurizer_stub("denseDrug", side="drug")
    register_matrix_predictor_stub("densePred")

    validate(_dense_model_config(sample_featurizer=FeaturizerConfig(name="cellBlocks", side="sample")))


def test_block_inputs_accept_a_featurizer_emitting_them() -> None:
    register_featurizer_stub("cellBlocks", side="sample", output_block_specs=_numeric("gene_expression"))
    register_featurizer_stub("denseDrug", side="drug")
    register_block_predictor_stub("blockPred").sample_inputs = Inputs.blocks(*_numeric("gene_expression"))

    validate(
        _dense_model_config(
            sample_featurizer=FeaturizerConfig(name="cellBlocks", side="sample"),
            predictor=PredictorConfig(name="blockPred"),
        )
    )


def test_block_inputs_name_the_missing_block_and_what_is_emitted() -> None:
    register_featurizer_stub("cellBlocks", side="sample", output_block_specs=_numeric("wrong_name"))
    register_featurizer_stub("denseDrug", side="drug")
    register_block_predictor_stub("blockPred").sample_inputs = Inputs.blocks(
        *_numeric("gene_expression"), *_numeric("tissue")
    )

    with pytest.raises(
        (ValueError, ValidationError),
        match=r"needs sample block\(s\) \['gene_expression', 'tissue'\].*emits \[wrong_name\]",
    ):
        ModelConfig(
            sample_featurizer=FeaturizerConfig(name="cellBlocks", side="sample"),
            drug_featurizer=FeaturizerConfig(name="denseDrug", side="drug"),
            predictor=PredictorConfig(name="blockPred"),
        )


def test_block_alternatives_need_one_of_them() -> None:
    register_featurizer_stub("cellBlocks", side="sample", output_block_specs=_numeric("tissue"))
    register_featurizer_stub("denseDrug", side="drug")
    alternatives = (*_numeric("gene_expression"), *_numeric("tissue"))
    register_block_predictor_stub("blockPred").sample_inputs = Inputs.blocks(alternatives)

    validate(
        _dense_model_config(
            sample_featurizer=FeaturizerConfig(name="cellBlocks", side="sample"),
            predictor=PredictorConfig(name="blockPred"),
        )
    )


def test_none_inputs_forbid_a_featurizer() -> None:
    register_featurizer_stub("denseSample", side="sample")
    register_feature_free_predictor_stub("naiveMean")

    with pytest.raises((ValueError, ValidationError), match="reads no sample features"):
        ModelConfig(
            sample_featurizer=FeaturizerConfig(name="denseSample", side="sample", view="gene_expression"),
            drug_featurizer=None,
            predictor=PredictorConfig(name="naiveMean"),
        )


def test_builtin_featurizer_declares_output_block_specs() -> None:
    from drevalpy.components.featurizers.drug.fingerprints import FingerprintsFeaturizer

    assert FingerprintsFeaturizer.output_blocks({}) == _numeric("fingerprints")


def test_feature_free_predictor_without_featurizers_passes() -> None:
    register_feature_free_predictor_stub("naiveMean")

    validate(ModelConfig(sample_featurizer=None, drug_featurizer=None, predictor=PredictorConfig(name="naiveMean")))


def test_feature_using_predictors_need_featurizers() -> None:
    register_dense_trio()
    with pytest.raises((ValueError, ValidationError), match="requires a sample_featurizer"):
        _dense_model_config(sample_featurizer=None, drug_featurizer=None)


def test_baseline_tag_does_not_allow_missing_featurizers() -> None:
    restore_builtin_components()
    with pytest.raises((ValueError, ValidationError), match="requires a sample_featurizer"):
        ModelConfig(sample_featurizer=None, drug_featurizer=None, predictor=PredictorConfig(name="naiveMeanEffects"))


def test_single_drug_scope_needs_no_drug_featurizer() -> None:
    restore_builtin_components()
    validate(
        ModelConfig(
            sample_featurizer=FeaturizerConfig(name="scaledGeneExpression", side="sample"),
            drug_featurizer=None,
            predictor=PredictorConfig(name="singleDrugElasticNet"),
        )
    )


def test_view_fallback_emits_named_block() -> None:
    register_featurizer_stub("viewCell", side="sample")

    config = FeaturizerConfig(name="viewCell", side="sample", view="gene_expression")

    assert config.output_blocks() == (BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),)


def test_declared_output_block_specs_win() -> None:
    register_featurizer_stub(
        "declaredDrug",
        side="drug",
        output_block_specs=(BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),),
    )

    config = FeaturizerConfig(name="declaredDrug", side="drug", view="ignored")

    assert config.output_blocks() == (BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),)


def test_nested_concat_flattens_child_blocks() -> None:
    from drevalpy.components.featurizers.shared.concat import SampleConcatFeaturizer

    register_featurizer_stub("denseSample", side="sample")
    featurizer_registry("sample").register(CONCAT, SampleConcatFeaturizer)

    config = FeaturizerConfig.parse(
        {
            "name": "concatFeaturizers",
            "featurizers": [
                {"name": "denseSample", "view": "gene_expression"},
                {
                    "name": "concatFeaturizers",
                    "featurizers": [
                        {"name": "denseSample", "view": "mutations"},
                    ],
                },
            ],
        },
        "sample",
    )
    assert config.output_blocks() == (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX),
    )


def test_sparsego_expression_and_mutations_block_names() -> None:
    restore_builtin_components()
    for default, block in (("expression", "gene_expression"), ("mutations", "mutations")):
        config = FeaturizerConfig(
            name="sparsegoOntology",
            side="sample",
            hyperparameter_space={
                "input_type": {"type": "categorical", "choices": ["expression", "mutations"], "default": default}
            },
        )
        assert config.output_blocks() == (BlockSpec(block, FeatureFormat.NUMERIC_MATRIX),)
