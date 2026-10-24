"""Tests for featurizer config → output block-spec resolution."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.models.config import DrugFeaturizerConfig, FeaturizerConfig, SampleFeaturizerConfig
from drevalpy.models.config._block_specs import resolve_output_block_specs
from drevalpy.registry.sample_featurizer import sample_featurizer_registry
from drevalpy.types.data.batch.feature_block import BlockSpec
from tests.models.config._stubs import register_featurizer_stub
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


def test_view_fallback_emits_named_block() -> None:
    register_featurizer_stub("viewCell", side="sample")

    config = SampleFeaturizerConfig(name="viewCell", view="gene_expression")

    assert resolve_output_block_specs(config) == (BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),)


def test_declared_output_block_specs_win() -> None:
    register_featurizer_stub(
        "declaredDrug",
        side="drug",
        output_block_specs=(BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),),
    )

    config = DrugFeaturizerConfig(name="declaredDrug", view="ignored")

    assert resolve_output_block_specs(config) == (BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),)


def test_nested_concat_flattens_child_blocks() -> None:
    from drevalpy.components.featurizers.shared.concat import SampleConcatFeaturizer

    register_featurizer_stub("denseSample", side="sample")
    sample_featurizer_registry.register_existing("concatFeaturizers", SampleConcatFeaturizer)

    config = SampleFeaturizerConfig.model_validate(
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
        }
    )
    assert resolve_output_block_specs(config) == (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX),
    )


def test_sparsego_expression_and_mutations_block_names() -> None:
    from drevalpy.registry._builtins import register_builtin_components

    register_builtin_components()
    for default, block in (("expression", "gene_expression"), ("mutations", "mutations")):
        config = FeaturizerConfig(
            name="sparsegoOntology",
            registry="sample",
            hyperparameter_space={
                "input_type": {"type": "categorical", "choices": ["expression", "mutations"], "default": default}
            },
        )
        assert resolve_output_block_specs(config) == (BlockSpec(block, FeatureFormat.NUMERIC_MATRIX, metadata=True),)
