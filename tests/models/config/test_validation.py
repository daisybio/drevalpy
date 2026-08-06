"""Tests for internal ModelConfig validation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import BlockSpec
from drevalpy.components.predictors.block import BlockPredictor
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.matrix import MatrixPredictor
from drevalpy.components.registry import (
    register_cell_line_featurizer,
    register_drug_featurizer,
    register_predictor,
)
from drevalpy.components.registry.featurizer_registry import (
    cell_line_featurizer_registry,
    drug_featurizer_registry,
)
from drevalpy.components.registry.predictor_registry import predictor_registry
from drevalpy.models.config import (
    CellLineFeaturizerConfig,
    DrugFeaturizerConfig,
    ModelConfig,
    ModelScope,
    PredictionMode,
    PredictorConfig,
)
from drevalpy.models.config.validation import validate


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    cell_line_featurizer_registry.clear()
    drug_featurizer_registry.clear()
    predictor_registry.clear()
    yield
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()


def _register_dense_pair() -> None:
    @register_cell_line_featurizer(
        "denseCellLine",
        description="dense cell line",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DenseCellLine:
        pass

    @register_drug_featurizer(
        "denseDrug",
        description="dense drug",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DenseDrug:
        pass

    @register_predictor(
        "densePred",
        description="dense pred",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DensePred(MatrixPredictor):
        supported_modes = frozenset({PredictionMode.REGRESSION})

        def _fit_matrix(self, x, y) -> None:
            return None

        def _predict_matrix(self, x):
            import numpy as np

            return np.zeros(len(x), dtype=np.float64)


def test_valid_dense_config_passes() -> None:
    _register_dense_pair()
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="denseCellLine", view="gene_expression"),
        drug_featurizer=DrugFeaturizerConfig(name="denseDrug", view="fingerprints"),
        predictor=PredictorConfig(name="densePred"),
    )
    validate(config)


def test_unknown_cell_line_featurizer_fails() -> None:
    _register_dense_pair()
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="missing"),
        drug_featurizer=DrugFeaturizerConfig(name="denseDrug"),
        predictor=PredictorConfig(name="densePred"),
    )
    with pytest.raises(ValueError, match="Unknown Cell line featurizer"):
        validate(config)


def test_wrong_registry_is_coerced_by_slot_subclasses() -> None:
    _register_dense_pair()
    config = ModelConfig.model_validate(
        {
            "cell_line_featurizer": {"name": "denseCellLine", "registry": "drug"},
            "drug_featurizer": {"name": "denseDrug", "registry": "cell_line"},
            "predictor": {"name": "densePred"},
        }
    )
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.registry == "cell_line"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.registry == "drug"
    validate(config)


def test_graph_featurizer_with_matrix_predictor_fails() -> None:
    @register_cell_line_featurizer(
        "graphCellLine",
        description="graph",
        contract=FeatureFormat.GRAPH,
    )
    class GraphCellLine:
        pass

    @register_drug_featurizer(
        "denseDrug",
        description="dense drug",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DenseDrug:
        pass

    @register_predictor(
        "densePred",
        description="dense pred",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DensePred(MatrixPredictor):
        supported_modes = frozenset({PredictionMode.REGRESSION})

        def _fit_matrix(self, x, y) -> None:
            return None

        def _predict_matrix(self, x):
            import numpy as np

            return np.zeros(len(x), dtype=np.float64)

    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="graphCellLine"),
        drug_featurizer=DrugFeaturizerConfig(name="denseDrug"),
        predictor=PredictorConfig(name="densePred"),
    )
    with pytest.raises(ValueError, match="Cell line featurizer contract|numeric_matrix"):
        validate(config)


def test_graph_format_match_passes_for_block_predictor() -> None:
    @register_cell_line_featurizer(
        "graphCellLine",
        description="graph",
        contract=FeatureFormat.GRAPH,
    )
    class GraphCellLine:
        pass

    @register_drug_featurizer(
        "graphDrug",
        description="graph drug",
        contract=FeatureFormat.GRAPH,
    )
    class GraphDrug:
        pass

    @register_predictor(
        "graphPred",
        description="graph pred",
        cell_line_contract=FeatureFormat.GRAPH,
        drug_contract=FeatureFormat.GRAPH,
    )
    class GraphPred(BlockPredictor):
        supported_modes = frozenset({PredictionMode.REGRESSION})

    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="graphCellLine"),
        drug_featurizer=DrugFeaturizerConfig(name="graphDrug"),
        predictor=PredictorConfig(name="graphPred"),
    )
    validate(config)


def test_block_schema_reports_missing_named_block() -> None:
    @register_cell_line_featurizer("cellBlocks", description="cell blocks", contract=FeatureFormat.NUMERIC_MATRIX)
    class CellBlocks:
        output_block_specs = (BlockSpec("wrong_name", FeatureFormat.NUMERIC_MATRIX),)

    @register_drug_featurizer("drugBlocks", description="drug blocks", contract=FeatureFormat.NUMERIC_MATRIX)
    class DrugBlocks:
        output_block_specs = (BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),)

    @register_predictor(
        "blockPred",
        description="block pred",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class BlockPred(BlockPredictor):
        required_cell_line_block_specs = (BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),)
        required_drug_block_specs = (BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),)

    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="cellBlocks"),
        drug_featurizer=DrugFeaturizerConfig(name="drugBlocks"),
        predictor=PredictorConfig(name="blockPred"),
    )
    with pytest.raises(ValueError, match="blockPred.*gene_expression.*numeric_matrix.*wrong_name"):
        validate(config)


def test_builtin_featurizer_declares_output_block_specs() -> None:
    from drevalpy.components.featurizers.drug.fingerprints import FingerprintsFeaturizer

    assert FingerprintsFeaturizer.output_block_specs == (BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),)


def test_feature_free_predictor_without_featurizers_passes() -> None:
    @register_predictor(
        "naiveMean",
        description="naive",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class NaiveMean(FeatureFreePredictor):
        pass

    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name="naiveMean"),
    )
    validate(config)


def test_feature_using_predictor_without_featurizers_fails() -> None:
    _register_dense_pair()
    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name="densePred"),
    )
    with pytest.raises(ValueError, match="requires featurizers"):
        validate(config)


def test_baseline_tag_does_not_allow_missing_featurizers() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name="naiveMeanEffects"),
    )
    with pytest.raises(ValueError, match="requires featurizers"):
        validate(config)


def test_empty_view_string_fails() -> None:
    _register_dense_pair()
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="denseCellLine", view="   "),
        drug_featurizer=DrugFeaturizerConfig(name="denseDrug"),
        predictor=PredictorConfig(name="densePred"),
    )
    with pytest.raises(ValueError, match="cell_line_featurizer view must be a non-empty string"):
        validate(config)


def test_scope_must_match_predictor_capability() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
        drug_featurizer=None,
        predictor=PredictorConfig(name="singleDrugElasticNet"),
        scope=ModelScope.MULTI_DRUG,
    )
    with pytest.raises(ValueError, match="does not support scope"):
        validate(config)


def test_single_drug_scope_requires_identity_drug_featurizer() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(name="singleDrugElasticNet"),
        scope=ModelScope.SINGLE_DRUG,
    )
    with pytest.raises(ValueError, match="requires drug_featurizer='identity'"):
        validate(config)


def test_single_drug_scope_accepts_identity_routing_featurizer() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
        drug_featurizer=DrugFeaturizerConfig(name="identity"),
        predictor=PredictorConfig(name="singleDrugElasticNet"),
        scope=ModelScope.SINGLE_DRUG,
    )
    validate(config)
