"""Tests for ModelConfig construction-time validation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from pydantic import ValidationError

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.models.config import (
    CellLineFeaturizerConfig,
    DrugFeaturizerConfig,
    ModelConfig,
    PredictionMode,
    PredictorConfig,
    validate,
)
from drevalpy.registry.cell_line_featurizer import register as register_cell_line_featurizer
from drevalpy.registry.drug_featurizer import register as register_drug_featurizer
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.types.data.batch.feature_block import BlockSpec
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


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
    with pytest.raises((ValueError, ValidationError), match="Unknown Cell line featurizer"):
        ModelConfig(
            cell_line_featurizer=CellLineFeaturizerConfig(name="missing"),
            drug_featurizer=DrugFeaturizerConfig(name="denseDrug"),
            predictor=PredictorConfig(name="densePred"),
        )


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

    with pytest.raises((ValueError, ValidationError), match="Cell line featurizer contract|numeric_matrix"):
        ModelConfig(
            cell_line_featurizer=CellLineFeaturizerConfig(name="graphCellLine"),
            drug_featurizer=DrugFeaturizerConfig(name="denseDrug"),
            predictor=PredictorConfig(name="densePred"),
        )


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

        def _fit(self, batch) -> None:
            return None

        def _predict(self, batch):
            return np.zeros(batch.n_pairs, dtype=np.float64)

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

        def _fit(self, batch) -> None:
            return None

        def _predict(self, batch):
            return np.zeros(batch.n_pairs, dtype=np.float64)

    with pytest.raises((ValueError, ValidationError), match="blockPred.*gene_expression.*numeric_matrix.*wrong_name"):
        ModelConfig(
            cell_line_featurizer=CellLineFeaturizerConfig(name="cellBlocks"),
            drug_featurizer=DrugFeaturizerConfig(name="drugBlocks"),
            predictor=PredictorConfig(name="blockPred"),
        )


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
        def _fit(self, batch) -> None:
            return None

        def _predict(self, batch):
            return np.zeros(batch.n_pairs, dtype=np.float64)

    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name="naiveMean"),
    )
    validate(config)


def test_feature_using_predictor_without_featurizers_fails() -> None:
    _register_dense_pair()
    with pytest.raises((ValueError, ValidationError), match="requires featurizers"):
        ModelConfig(
            cell_line_featurizer=None,
            drug_featurizer=None,
            predictor=PredictorConfig(name="densePred"),
        )


def test_baseline_tag_does_not_allow_missing_featurizers() -> None:
    from drevalpy.registry._builtins import register_builtin_components

    register_builtin_components()
    with pytest.raises((ValueError, ValidationError), match="requires featurizers"):
        ModelConfig(
            cell_line_featurizer=None,
            drug_featurizer=None,
            predictor=PredictorConfig(name="naiveMeanEffects"),
        )


def test_single_drug_scope_requires_identity_drug_featurizer() -> None:
    from drevalpy.registry._builtins import register_builtin_components

    register_builtin_components()
    with pytest.raises((ValueError, ValidationError), match="requires drug_featurizer='identity'"):
        ModelConfig(
            cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
            drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
            predictor=PredictorConfig(name="singleDrugElasticNet"),
        )


def test_single_drug_scope_accepts_identity_routing_featurizer() -> None:
    from drevalpy.registry._builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
        drug_featurizer=DrugFeaturizerConfig(name="identity"),
        predictor=PredictorConfig(name="singleDrugElasticNet"),
    )
    validate(config)
