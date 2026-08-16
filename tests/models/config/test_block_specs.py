"""Tests for featurizer config → output block-spec resolution."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.models.config import CellLineFeaturizerConfig, DrugFeaturizerConfig, FeaturizerConfig
from drevalpy.models.config._block_specs import resolve_output_block_specs
from drevalpy.registry.cell_line_featurizer import cell_line_featurizer_registry
from drevalpy.types.data.batch.feature_block import BlockSpec
from tests.models.config._stubs import register_featurizer_stub
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


def test_view_fallback_emits_named_block() -> None:
    register_featurizer_stub("viewCell", side="cell_line")

    config = CellLineFeaturizerConfig(name="viewCell", view="gene_expression")

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
    from drevalpy.components.featurizers.shared.concat import CellLineConcatFeaturizer

    register_featurizer_stub("denseCellLine", side="cell_line")
    cell_line_featurizer_registry.register_existing("concatFeaturizers", CellLineConcatFeaturizer)

    config = CellLineFeaturizerConfig.model_validate(
        {
            "name": "concatFeaturizers",
            "featurizers": [
                {"name": "denseCellLine", "view": "gene_expression"},
                {
                    "name": "concatFeaturizers",
                    "featurizers": [
                        {"name": "denseCellLine", "view": "mutations"},
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
    expression = FeaturizerConfig(
        name="sparsegoOntology",
        registry="cell_line",
        hyperparameter_space={
            "input_type": {"type": "categorical", "choices": ["expression", "mutations"], "default": "expression"}
        },
    )
    mutations = FeaturizerConfig(
        name="sparsegoOntology",
        registry="cell_line",
        hyperparameter_space={
            "input_type": {"type": "categorical", "choices": ["expression", "mutations"], "default": "mutations"}
        },
    )
    assert resolve_output_block_specs(expression) == (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX, metadata=True),
    )
    assert resolve_output_block_specs(mutations) == (
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX, metadata=True),
    )
