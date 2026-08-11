"""Tests for featurizer config → output block-spec resolution."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.core.batch.feature_block import BlockSpec
from drevalpy.components.registry import register_cell_line_featurizer, register_drug_featurizer
from drevalpy.components.registry.featurizer_registry import (
    cell_line_featurizer_registry,
    drug_featurizer_registry,
)
from drevalpy.components.registry.predictor_registry import predictor_registry
from drevalpy.models.config import CellLineFeaturizerConfig, DrugFeaturizerConfig, FeaturizerConfig
from drevalpy.models.config._block_specs import resolve_output_block_specs


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    cell_line_featurizer_registry.clear()
    drug_featurizer_registry.clear()
    predictor_registry.clear()
    yield
    from drevalpy.components.registry.register_builtins import register_builtin_components

    register_builtin_components()


def test_view_fallback_emits_named_block() -> None:
    @register_cell_line_featurizer(
        "viewCell",
        description="view cell",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class ViewCell:
        pass

    config = CellLineFeaturizerConfig(name="viewCell", view="gene_expression")
    assert resolve_output_block_specs(config) == (BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),)


def test_declared_output_block_specs_win() -> None:
    @register_drug_featurizer(
        "declaredDrug",
        description="declared",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DeclaredDrug:
        output_block_specs = (BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),)

    config = DrugFeaturizerConfig(name="declaredDrug", view="ignored")
    assert resolve_output_block_specs(config) == (BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),)


def test_nested_concat_flattens_child_blocks() -> None:
    from drevalpy.components.featurizers.cell_line.concat import ConcatFeaturizersCellLineFeaturizer

    @register_cell_line_featurizer(
        "denseCellLine",
        description="dense",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DenseCellLine:
        pass

    cell_line_featurizer_registry.register_existing("concatFeaturizers", ConcatFeaturizersCellLineFeaturizer)

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
    from drevalpy.components.registry.register_builtins import register_builtin_components

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
