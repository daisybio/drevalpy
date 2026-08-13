"""Tests for hyperparameter-space default enforcement."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.contracts.hyperparameter_space import (
    validate_component_hyperparameter_space,
    validate_hyperparameter_space,
)
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.models.config import FeaturizerConfig, PredictorConfig
from drevalpy.registry.cell_line_featurizer import cell_line_featurizer_registry
from drevalpy.registry.cell_line_featurizer import register as register_cell_line_featurizer
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.predictor import predictor_registry
from drevalpy.registry.predictor import register as register_predictor


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    cell_line_featurizer_registry.clear()
    drug_featurizer_registry.clear()
    predictor_registry.clear()
    yield
    from drevalpy.registry._builtins import register_builtin_components

    register_builtin_components()


def test_validate_hyperparameter_space_rejects_missing_default() -> None:
    with pytest.raises(ValueError, match="missing 'default' for 'alpha'"):
        validate_hyperparameter_space(
            {"alpha": {"type": "float", "low": 0.1, "high": 1.0}},
            context="unit-test",
        )


def test_validate_hyperparameter_space_rejects_non_mapping_spec() -> None:
    with pytest.raises(ValueError, match="non-mapping specs for 'alpha'"):
        validate_hyperparameter_space({"alpha": 1.0}, context="unit-test")


def test_validate_hyperparameter_space_accepts_complete_specs() -> None:
    validate_hyperparameter_space(
        {"alpha": {"type": "float", "default": 0.5}},
        context="unit-test",
    )


def test_predictor_registration_rejects_space_without_default() -> None:
    with pytest.raises(ValueError, match="missing 'default'"):

        @register_predictor(
            "badSpacePred",
            description="missing default",
            cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class BadSpacePred(MatrixPredictor):
            @classmethod
            def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
                return {"alpha": {"type": "float", "low": 0.1, "high": 1.0}}

            def _fit_matrix(self, x, y) -> None:
                return None

            def _predict_matrix(self, x):
                return x[:, 0]


def test_featurizer_registration_rejects_space_without_default() -> None:
    with pytest.raises(ValueError, match="missing 'default'"):

        @register_cell_line_featurizer(
            "badSpaceFeat",
            description="missing default",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class BadSpaceFeat:
            @classmethod
            def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
                return {"n_components": {"type": "int", "low": 8, "high": 64}}


def test_predictor_config_rejects_space_without_default() -> None:
    with pytest.raises(ValidationError, match="missing 'default'"):
        PredictorConfig(
            name="elasticNet",
            hyperparameter_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0}},
        )


def test_featurizer_config_rejects_space_without_default() -> None:
    with pytest.raises(ValidationError, match="missing 'default'"):
        FeaturizerConfig(
            name="pca",
            view="expression",
            hyperparameter_space={"n_components": {"type": "int", "low": 8, "high": 64}},
        )


def test_validate_component_hyperparameter_space_uses_class_getter() -> None:
    class Ok:
        @classmethod
        def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
            return {"alpha": {"type": "float", "default": 1.0}}

    validate_component_hyperparameter_space("ok", Ok)
