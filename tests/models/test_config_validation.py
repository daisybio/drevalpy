"""Tests for internal ModelConfig validation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.matrix import MatrixPredictor
from drevalpy.components.predictors.structured import BlockPredictor
from drevalpy.components.registry import (
    clear_cell_line_featurizer_registry,
    clear_drug_featurizer_registry,
    clear_predictor_registry,
    register_cell_line_featurizer,
    register_drug_featurizer,
    register_predictor,
)
from drevalpy.models.config import (
    FeaturizerConfig,
    ModelConfig,
    ModelScope,
    PredictionMode,
    PredictorConfig,
)
from drevalpy.models.config_validation import validate_model_config


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    clear_cell_line_featurizer_registry()
    clear_drug_featurizer_registry()
    clear_predictor_registry()
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
        cell_line_featurizer=FeaturizerConfig(name="denseCellLine", registry="cell_line", view="gene_expression"),
        drug_featurizer=FeaturizerConfig(name="denseDrug", registry="drug", view="fingerprints"),
        predictor=PredictorConfig(name="densePred"),
    )
    validate_model_config(config)


def test_unknown_cell_line_featurizer_fails() -> None:
    _register_dense_pair()
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="missing", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="denseDrug", registry="drug"),
        predictor=PredictorConfig(name="densePred"),
    )
    with pytest.raises(ValueError, match="Unknown Cell line featurizer"):
        validate_model_config(config)


def test_wrong_registry_slot_fails() -> None:
    _register_dense_pair()
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="denseCellLine", registry="drug"),
        drug_featurizer=FeaturizerConfig(name="denseDrug", registry="drug"),
        predictor=PredictorConfig(name="densePred"),
    )
    with pytest.raises(ValueError, match="cell_line_featurizer must use registry='cell_line'"):
        validate_model_config(config)


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
        cell_line_featurizer=FeaturizerConfig(name="graphCellLine", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="denseDrug", registry="drug"),
        predictor=PredictorConfig(name="densePred"),
    )
    with pytest.raises(ValueError, match="Cell line featurizer contract|numeric_matrix"):
        validate_model_config(config)


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
        cell_line_featurizer=FeaturizerConfig(name="graphCellLine", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="graphDrug", registry="drug"),
        predictor=PredictorConfig(name="graphPred"),
    )
    validate_model_config(config)


def test_feature_free_predictor_without_featurizers_passes() -> None:
    @register_predictor("naiveMean", description="naive")
    class NaiveMean(FeatureFreePredictor):
        pass

    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name="naiveMean"),
    )
    validate_model_config(config)


def test_feature_using_predictor_without_featurizers_fails() -> None:
    _register_dense_pair()
    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name="densePred"),
    )
    with pytest.raises(ValueError, match="requires featurizers"):
        validate_model_config(config)


def test_baseline_tag_does_not_allow_missing_featurizers() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name="naiveMeanEffects"),
    )
    with pytest.raises(ValueError, match="requires featurizers"):
        validate_model_config(config)


def test_empty_view_string_fails() -> None:
    _register_dense_pair()
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="denseCellLine", registry="cell_line", view="   "),
        drug_featurizer=FeaturizerConfig(name="denseDrug", registry="drug"),
        predictor=PredictorConfig(name="densePred"),
    )
    with pytest.raises(ValueError, match="cell_line_featurizer view must be a non-empty string"):
        validate_model_config(config)


def test_scope_must_match_predictor_capability() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="scaledGeneExpression", registry="cell_line"),
        drug_featurizer=None,
        predictor=PredictorConfig(name="singleDrugElasticNet"),
        scope=ModelScope.MULTI_DRUG,
    )
    with pytest.raises(ValueError, match="does not support scope"):
        validate_model_config(config)


def test_single_drug_scope_requires_identity_drug_featurizer() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="scaledGeneExpression", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="fingerprints", registry="drug"),
        predictor=PredictorConfig(name="singleDrugElasticNet"),
        scope=ModelScope.SINGLE_DRUG,
    )
    with pytest.raises(ValueError, match="requires drug_featurizer='identity'"):
        validate_model_config(config)


def test_single_drug_scope_accepts_identity_routing_featurizer() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="scaledGeneExpression", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="identity", registry="drug"),
        predictor=PredictorConfig(name="singleDrugElasticNet"),
        scope=ModelScope.SINGLE_DRUG,
    )
    validate_model_config(config)
