"""Tests for internal ModelConfig validation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts import FeatureKind
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
        category="native",
        contract=FeatureKind.DENSE,
    )
    class DenseCellLine:
        pass

    @register_drug_featurizer(
        "denseDrug",
        description="dense drug",
        category="native",
        contract=FeatureKind.DENSE,
    )
    class DenseDrug:
        pass

    @register_predictor(
        "densePred",
        description="dense pred",
        category="general_purpose",
        cell_line_contract=FeatureKind.DENSE,
        drug_contract=FeatureKind.DENSE,
    )
    class DensePred:
        supported_modes = {PredictionMode.REGRESSION}


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


def test_graph_featurizer_with_dense_predictor_fails() -> None:
    @register_cell_line_featurizer(
        "graphCellLine",
        description="graph",
        category="native",
        contract=FeatureKind.GRAPH,
    )
    class GraphCellLine:
        pass

    @register_drug_featurizer(
        "denseDrug",
        description="dense drug",
        category="native",
        contract=FeatureKind.DENSE,
    )
    class DenseDrug:
        pass

    @register_predictor(
        "densePred",
        description="dense pred",
        category="general_purpose",
        cell_line_contract=FeatureKind.DENSE,
        drug_contract=FeatureKind.DENSE,
    )
    class DensePred:
        supported_modes = {PredictionMode.REGRESSION}

    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="graphCellLine", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="denseDrug", registry="drug"),
        predictor=PredictorConfig(name="densePred"),
    )
    with pytest.raises(ValueError, match="Cell line featurizer contract"):
        validate_model_config(config)


def test_graph_kind_match_passes() -> None:
    @register_cell_line_featurizer(
        "graphCellLine",
        description="graph",
        category="native",
        contract=FeatureKind.GRAPH,
    )
    class GraphCellLine:
        pass

    @register_drug_featurizer(
        "graphDrug",
        description="graph drug",
        category="native",
        contract=FeatureKind.GRAPH,
    )
    class GraphDrug:
        pass

    @register_predictor(
        "graphPred",
        description="graph pred",
        category="general_purpose",
        cell_line_contract=FeatureKind.GRAPH,
        drug_contract=FeatureKind.GRAPH,
    )
    class GraphPred:
        supported_modes = {PredictionMode.REGRESSION}

    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="graphCellLine", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="graphDrug", registry="drug"),
        predictor=PredictorConfig(name="graphPred"),
    )
    validate_model_config(config)


def test_feature_free_predictor_without_featurizers_passes() -> None:
    @register_predictor("naiveMean", description="naive", category="baseline")
    class NaiveMean:
        supported_modes = {PredictionMode.REGRESSION}

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


def test_single_drug_scope_forbids_drug_featurizer() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="scaledGeneExpression", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="fingerprints", registry="drug"),
        predictor=PredictorConfig(name="singleDrugElasticNet"),
        scope=ModelScope.SINGLE_DRUG,
    )
    with pytest.raises(ValueError, match="forbids a drug_featurizer"):
        validate_model_config(config)
