"""Validation logic for `~drevalpy.models.config.ModelConfig`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from drevalpy.components.contracts import (
    FeatureContract,
    contracts_compatible,
    featurizer_contract,
    predictor_contracts,
)
from drevalpy.components.predictors.baseline import BaselinePredictor
from drevalpy.components.registry import lookup as _registry_lookup

if TYPE_CHECKING:
    from drevalpy.models.config import FeaturizerConfig, ModelConfig


def _validate_view_fields(featurizer: FeaturizerConfig, *, label: str) -> None:
    if featurizer.view is not None and not str(featurizer.view).strip():
        msg = f"{label} view must be a non-empty string when set"
        raise ValueError(msg)
    if featurizer.views is not None:
        if not featurizer.views:
            msg = f"{label} views must be a non-empty list when set"
            raise ValueError(msg)
        if any(not str(view).strip() for view in featurizer.views):
            msg = f"{label} views must contain non-empty strings"
            raise ValueError(msg)


def _featurizer_contract(cls: type[Any]) -> FeatureContract:
    try:
        return featurizer_contract(cls)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def _predictor_contracts(cls: type[Any]) -> tuple[FeatureContract, FeatureContract]:
    try:
        return predictor_contracts(cls)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def _allows_no_featurizers(pred_cls: type[Any]) -> bool:
    if issubclass(pred_cls, BaselinePredictor):
        return True
    return getattr(pred_cls, "category", "") == "baseline"


def validate_model_config(config: ModelConfig) -> None:
    """Check registry slots, feature compatibility, and prediction mode."""
    pred_cls = _registry_lookup.get_predictor(config.predictor.name)

    if config.cell_line_featurizer is None and config.drug_featurizer is None:
        if not _allows_no_featurizers(pred_cls):
            msg = (
                f"Predictor {config.predictor.name!r} requires featurizers; "
                "set cell_line_featurizer and drug_featurizer, or use a baseline predictor."
            )
            raise ValueError(msg)
        supported = getattr(pred_cls, "supported_modes", None)
        if supported is not None and config.prediction_mode not in supported:
            msg = (
                f"Predictor {config.predictor.name!r} does not support "
                f"prediction_mode={config.prediction_mode!r}; "
                f"supported_modes={sorted(supported)}"
            )
            raise ValueError(msg)
        return

    requires_drug = getattr(pred_cls, "requires_drug_featurizer", True)

    if requires_drug and config.drug_featurizer is None:
        msg = f"Predictor {config.predictor.name!r} requires a drug_featurizer"
        raise ValueError(msg)

    if config.cell_line_featurizer is None and not _allows_no_featurizers(pred_cls):
        msg = "cell_line_featurizer must be set for feature-based predictors"
        raise ValueError(msg)

    if config.cell_line_featurizer is not None and config.cell_line_featurizer.registry != "cell_line":
        msg = f"cell_line_featurizer must use registry='cell_line', got {config.cell_line_featurizer.registry!r}"
        raise ValueError(msg)
    if config.drug_featurizer is not None and config.drug_featurizer.registry != "drug":
        msg = "drug_featurizer must use registry='drug', " f"got {config.drug_featurizer.registry!r}"
        raise ValueError(msg)

    if config.cell_line_featurizer is not None:
        _validate_view_fields(config.cell_line_featurizer, label="cell_line_featurizer")
    if config.drug_featurizer is not None:
        _validate_view_fields(config.drug_featurizer, label="drug_featurizer")

    supported = getattr(pred_cls, "supported_modes", None)
    if supported is not None and config.prediction_mode not in supported:
        msg = (
            f"Predictor {config.predictor.name!r} does not support "
            f"prediction_mode={config.prediction_mode!r}; "
            f"supported_modes={sorted(supported)}"
        )
        raise ValueError(msg)

    required_cell_line, required_drug = _predictor_contracts(pred_cls)

    if config.cell_line_featurizer is not None:
        cell_line_cls = _registry_lookup.get_cell_line_featurizer(config.cell_line_featurizer.name)
        cell_line_contract = _featurizer_contract(cell_line_cls)
        if not contracts_compatible(cell_line_contract, required_cell_line):
            msg = (
                f"Cell line featurizer contract {cell_line_contract!r} is incompatible with "
                f"predictor cell_line_contract {required_cell_line!r}"
            )
            raise ValueError(msg)

    if config.drug_featurizer is not None:
        drug_cls = _registry_lookup.get_drug_featurizer(config.drug_featurizer.name)
        drug_contract = _featurizer_contract(drug_cls)
        if not contracts_compatible(drug_contract, required_drug):
            msg = (
                f"Drug featurizer contract {drug_contract!r} is incompatible "
                f"with predictor drug_contract {required_drug!r}"
            )
            raise ValueError(msg)
