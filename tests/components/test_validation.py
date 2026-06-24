"""Tests for internal ModelConfig validation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.config import (
    FeaturizerConfig,
    ModelConfig,
    PredictionMode,
    PredictorConfig,
)
from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.registry import (
    clear_cell_line_featurizer_registry,
    clear_drug_featurizer_registry,
    clear_predictor_registry,
    register_cell_line_featurizer,
    register_drug_featurizer,
    register_predictor,
)
from drevalpy.components.validation import validate_model_config


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    clear_cell_line_featurizer_registry()
    clear_drug_featurizer_registry()
    clear_predictor_registry()
    yield
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()


def _register_dense_pair() -> None:
    @register_cell_line_featurizer("denseCellLine", description="dense cell line", category="native")
    class DenseCellLine:
        output_contract = FeatureContract(kind=FeatureKind.DENSE)

    @register_drug_featurizer("denseDrug", description="dense drug", category="native")
    class DenseDrug:
        output_contract = FeatureContract(kind=FeatureKind.DENSE)

    @register_predictor("densePred", description="dense pred", category="general_purpose")
    class DensePred:
        uses_features = True
        supported_modes = {PredictionMode.REGRESSION}
        required_cell_line_contract = FeatureContract(kind=FeatureKind.DENSE)
        required_drug_contract = FeatureContract(kind=FeatureKind.DENSE)


def test_valid_dense_config_passes() -> None:
    _register_dense_pair()
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="denseCellLine", registry="cell_line", view="gene_expression"),
        drug_featurizer=FeaturizerConfig(name="denseDrug", registry="drug", view="fingerprints"),
        predictor=PredictorConfig(type="densePred"),
    )
    validate_model_config(config)


def test_unknown_cell_line_featurizer_fails() -> None:
    _register_dense_pair()
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="missing", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="denseDrug", registry="drug"),
        predictor=PredictorConfig(type="densePred"),
    )
    with pytest.raises(ValueError, match="Unknown Cell line featurizer"):
        validate_model_config(config)


def test_wrong_registry_slot_fails() -> None:
    _register_dense_pair()
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="denseCellLine", registry="drug"),
        drug_featurizer=FeaturizerConfig(name="denseDrug", registry="drug"),
        predictor=PredictorConfig(type="densePred"),
    )
    with pytest.raises(ValueError, match="cell_line_featurizer must use registry='cell_line'"):
        validate_model_config(config)


def test_graph_featurizer_with_dense_predictor_fails() -> None:
    @register_cell_line_featurizer("graphCellLine", description="graph", category="native")
    class GraphCellLine:
        output_contract = FeatureContract(kind=FeatureKind.GRAPH, backend="pyg")

    @register_drug_featurizer("denseDrug", description="dense drug", category="native")
    class DenseDrug:
        output_contract = FeatureContract(kind=FeatureKind.DENSE)

    @register_predictor("densePred", description="dense pred", category="general_purpose")
    class DensePred:
        uses_features = True
        supported_modes = {PredictionMode.REGRESSION}
        required_cell_line_contract = FeatureContract(kind=FeatureKind.DENSE)
        required_drug_contract = FeatureContract(kind=FeatureKind.DENSE)

    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="graphCellLine", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="denseDrug", registry="drug"),
        predictor=PredictorConfig(type="densePred"),
    )
    with pytest.raises(ValueError, match="Cell line featurizer output_contract"):
        validate_model_config(config)


def test_graph_backend_mismatch_fails() -> None:
    @register_cell_line_featurizer("graphCellLine", description="graph", category="native")
    class GraphCellLine:
        output_contract = FeatureContract(kind=FeatureKind.GRAPH, backend="pyg")

    @register_drug_featurizer("graphDrug", description="graph drug", category="native")
    class GraphDrug:
        output_contract = FeatureContract(kind=FeatureKind.GRAPH, backend="pyg")

    @register_predictor("graphPred", description="graph pred", category="general_purpose")
    class GraphPred:
        uses_features = True
        supported_modes = {PredictionMode.REGRESSION}
        required_cell_line_contract = FeatureContract(kind=FeatureKind.GRAPH, backend="dgl")
        required_drug_contract = FeatureContract(kind=FeatureKind.GRAPH, backend="pyg")

    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="graphCellLine", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="graphDrug", registry="drug"),
        predictor=PredictorConfig(type="graphPred"),
    )
    with pytest.raises(ValueError, match="Cell line featurizer output_contract"):
        validate_model_config(config)


def test_feature_free_predictor_without_featurizers_passes() -> None:
    @register_predictor("naiveMean", description="naive", category="baseline")
    class NaiveMean:
        uses_features = False
        supported_modes = {PredictionMode.REGRESSION}

    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(type="naiveMean"),
    )
    validate_model_config(config)


def test_feature_using_predictor_without_featurizers_fails() -> None:
    _register_dense_pair()
    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(type="densePred"),
    )
    with pytest.raises(ValueError, match="uses feature matrices"):
        validate_model_config(config)


def test_empty_view_string_fails() -> None:
    _register_dense_pair()
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="denseCellLine", registry="cell_line", view="   "),
        drug_featurizer=FeaturizerConfig(name="denseDrug", registry="drug"),
        predictor=PredictorConfig(type="densePred"),
    )
    with pytest.raises(ValueError, match="cell_line_featurizer view must be a non-empty string"):
        validate_model_config(config)
