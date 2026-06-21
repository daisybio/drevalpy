"""Validation logic for :class:`~drevalpy.components.config.ModelConfig`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from drevalpy.components.contracts import FeatureContract, contracts_compatible
from drevalpy.components.registry import lookup as _registry_lookup

if TYPE_CHECKING:
    from drevalpy.components.config import FeaturizerConfig, ModelConfig


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
    contract = getattr(cls, "output_contract", None)
    if contract is None:
        msg = f"Featurizer {cls.__name__!r} must define output_contract"
        raise ValueError(msg)
    if not isinstance(contract, FeatureContract):
        msg = f"Featurizer {cls.__name__!r} output_contract must be a FeatureContract"
        raise ValueError(msg)
    return contract


def _predictor_contracts(cls: type[Any]) -> tuple[FeatureContract, FeatureContract]:
    cell_line = getattr(cls, "required_cell_line_contract", None)
    drug = getattr(cls, "required_drug_contract", None)
    if cell_line is None or drug is None:
        msg = (
            f"Predictor {cls.__name__!r} must define required_cell_line_contract "
            "and required_drug_contract"
        )
        raise ValueError(msg)
    if not isinstance(cell_line, FeatureContract) or not isinstance(drug, FeatureContract):
        msg = f"Predictor {cls.__name__!r} contracts must be FeatureContract instances"
        raise ValueError(msg)
    return cell_line, drug


def validate_model_config(config: ModelConfig) -> None:
    """Check registry slots, feature compatibility, and prediction mode."""
    if config.cell_line_featurizer is None and config.drug_featurizer is None:
        pred_cls = _registry_lookup.get_predictor(config.predictor.type)
        if getattr(pred_cls, "uses_features", True):
            msg = (
                f"Predictor {config.predictor.type!r} uses feature matrices; "
                "set cell_line_featurizer and drug_featurizer, or use a baseline predictor "
                "(uses_features=False)."
            )
            raise ValueError(msg)
        supported = getattr(pred_cls, "supported_modes", None)
        if supported is not None and config.prediction_mode not in supported:
            msg = (
                f"Predictor {config.predictor.type!r} does not support "
                f"prediction_mode={config.prediction_mode!r}; "
                f"supported_modes={sorted(supported)}"
            )
            raise ValueError(msg)
        return

    if config.cell_line_featurizer is None and config.drug_featurizer is None:
        msg = "At least one featurizer must be set when using a feature-based predictor"
        raise ValueError(msg)

    if config.cell_line_featurizer is not None and config.cell_line_featurizer.registry != "cell_line":
        msg = f"cell_line_featurizer must use registry='cell_line', got {config.cell_line_featurizer.registry!r}"
        raise ValueError(msg)
    if config.drug_featurizer is not None and config.drug_featurizer.registry != "drug":
        msg = (
            "drug_featurizer must use registry='drug', "
            f"got {config.drug_featurizer.registry!r}"
        )
        raise ValueError(msg)

    if config.cell_line_featurizer is not None:
        _validate_view_fields(config.cell_line_featurizer, label="cell_line_featurizer")
    if config.drug_featurizer is not None:
        _validate_view_fields(config.drug_featurizer, label="drug_featurizer")

    pred_cls = _registry_lookup.get_predictor(config.predictor.type)

    supported = getattr(pred_cls, "supported_modes", None)
    if supported is not None and config.prediction_mode not in supported:
        msg = (
            f"Predictor {config.predictor.type!r} does not support "
            f"prediction_mode={config.prediction_mode!r}; "
            f"supported_modes={sorted(supported)}"
        )
        raise ValueError(msg)

    required_cell_line, required_drug = _predictor_contracts(pred_cls)

    if config.cell_line_featurizer is not None:
        cell_line_cls = _registry_lookup.get_cell_line_featurizer(config.cell_line_featurizer.type)
        cell_line_contract = _featurizer_contract(cell_line_cls)
        if not contracts_compatible(cell_line_contract, required_cell_line):
            msg = (
                f"Cell line featurizer output_contract {cell_line_contract!r} is incompatible with "
                f"predictor required_cell_line_contract {required_cell_line!r}"
            )
            raise ValueError(msg)

    if config.drug_featurizer is not None:
        drug_cls = _registry_lookup.get_drug_featurizer(config.drug_featurizer.type)
        drug_contract = _featurizer_contract(drug_cls)
        if not contracts_compatible(drug_contract, required_drug):
            msg = (
                f"Drug featurizer output_contract {drug_contract!r} is incompatible "
                f"with predictor required_drug_contract {required_drug!r}"
            )
            raise ValueError(msg)
