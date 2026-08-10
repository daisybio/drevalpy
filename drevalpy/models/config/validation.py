"""Validation logic for `~drevalpy.models.config.ModelConfig`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from drevalpy.components.core.batch.feature_block import BlockSpec
from drevalpy.components.core.contracts.contracts import contracts_compatible, featurizer_contract, predictor_contracts
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.single_drug_routing import ROUTING_DRUG_FEATURIZER
from drevalpy.components.registry import (
    get_cell_line_featurizer,
    get_drug_featurizer,
    get_predictor,
)
from drevalpy.models.config._block_specs import resolve_output_block_specs
from drevalpy.types.enums.model_scope import ModelScope

if TYPE_CHECKING:
    from drevalpy.models.config.model import ModelConfig


def _validate_prediction_mode(config: ModelConfig, pred_cls: type[Any]) -> None:
    supported = getattr(pred_cls, "supported_modes", None)
    if supported is not None and config.prediction_mode not in supported:
        msg = (
            f"Predictor {config.predictor.name!r} does not support "
            f"prediction_mode={config.prediction_mode!r}; "
            f"supported_modes={sorted(supported)}"
        )
        raise ValueError(msg)


def _validate_feature_free_config(config: ModelConfig) -> None:
    if config.cell_line_featurizer is not None or config.drug_featurizer is not None:
        msg = f"Predictor {config.predictor.name!r} is a feature-free predictor and forbids configured featurizers"
        raise ValueError(msg)


def _validate_featurizer_presence(config: ModelConfig) -> None:
    if config.cell_line_featurizer is None and config.drug_featurizer is None:
        msg = (
            f"Predictor {config.predictor.name!r} requires featurizers; "
            "set cell_line_featurizer and drug_featurizer, or use a feature-free predictor."
        )
        raise ValueError(msg)
    if config.drug_featurizer is None:
        msg = f"Predictor {config.predictor.name!r} requires a drug_featurizer"
        raise ValueError(msg)
    if config.cell_line_featurizer is None:
        msg = "cell_line_featurizer must be set for feature-based predictors"
        raise ValueError(msg)


def _validate_single_drug_pairing(config: ModelConfig, pred_cls: type[Any]) -> None:
    if pred_cls.scope != ModelScope.SINGLE_DRUG:
        return
    if config.drug_featurizer is None or config.drug_featurizer.name != ROUTING_DRUG_FEATURIZER:
        msg = (
            f"Feature-based single-drug predictor {config.predictor.name!r} requires "
            f"drug_featurizer={ROUTING_DRUG_FEATURIZER!r} for per-drug routing"
        )
        raise ValueError(msg)


def _validate_featurizer_contracts(config: ModelConfig, pred_cls: type[Any]) -> None:
    required_cell_line, required_drug = predictor_contracts(pred_cls)
    sides = (
        (
            "cell_line",
            config.cell_line_featurizer,
            required_cell_line,
            get_cell_line_featurizer,
            "Cell line",
        ),
        (
            "drug",
            config.drug_featurizer,
            required_drug,
            get_drug_featurizer,
            "Drug",
        ),
    )
    for side, featurizer, required, getter, label in sides:
        if featurizer is None:
            continue
        contract = featurizer_contract(getter(featurizer.name))
        if not contracts_compatible(contract, required):
            msg = (
                f"{label} featurizer contract {contract!r} is incompatible with predictor {side}_contract {required!r}"
            )
            raise ValueError(msg)


def _validate_required_block_specs(
    predictor_name: str,
    side: str,
    emitted: tuple[BlockSpec, ...],
    required: tuple[BlockSpec, ...],
) -> None:
    actual = ", ".join(f"{spec.name}:{spec.format.value}" for spec in emitted) or "<none>"
    for expected in required:
        actual_spec = next((spec for spec in emitted if spec.name == expected.name), None)
        if actual_spec is None or actual_spec.format != expected.format:
            actual_format = actual_spec.format.value if actual_spec is not None else "<missing>"
            raise ValueError(
                f"Predictor {predictor_name!r} {side} block schema mismatch: missing block "
                f"{expected.name!r}; expected format={expected.format.value!r}, "
                f"actual format={actual_format!r}; emitted blocks=[{actual}]"
            )


def _validate_block_schema(config: ModelConfig, pred_cls: type[Any]) -> None:
    if pred_cls.input_interface != "block":
        return
    for side, featurizer in (
        ("cell_line", config.cell_line_featurizer),
        ("drug", config.drug_featurizer),
    ):
        if featurizer is None:
            continue
        emitted = resolve_output_block_specs(featurizer)
        required = tuple(getattr(pred_cls, f"required_{side}_block_specs", ()))
        _validate_required_block_specs(config.predictor.name, side, emitted, required)
        alternatives = tuple(getattr(pred_cls, f"required_{side}_block_alternatives", ()))
        if alternatives and not any(
            actual.name == option.name and actual.format == option.format
            for actual in emitted
            for option in alternatives
        ):
            expected = " or ".join(f"{item.name}:{item.format.value}" for item in alternatives)
            actual = ", ".join(f"{item.name}:{item.format.value}" for item in emitted) or "<none>"
            raise ValueError(
                f"Predictor {config.predictor.name!r} {side} block schema mismatch: expected one of "
                f"[{expected}], actual emitted blocks=[{actual}]"
            )


def validate(config: ModelConfig) -> ModelConfig:
    """Check registry slots, feature compatibility, and prediction mode.

    :param config: Model configuration to validate.
    :returns: The unchanged *config*, so this doubles as a Pydantic ``after`` validator.
    """
    try:
        pred_cls = get_predictor(config.predictor.name)
    except ImportError:
        return config
    _validate_prediction_mode(config, pred_cls)
    if issubclass(pred_cls, FeatureFreePredictor):
        _validate_feature_free_config(config)
        return config
    _validate_featurizer_presence(config)
    _validate_single_drug_pairing(config, pred_cls)
    _validate_featurizer_contracts(config, pred_cls)
    _validate_block_schema(config, pred_cls)
    return config
