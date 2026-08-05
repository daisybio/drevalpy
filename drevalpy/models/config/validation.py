"""Validation logic for `~drevalpy.models.config.ModelConfig`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from drevalpy.components.contracts import (
    FeatureFormat,
    contracts_compatible,
    featurizer_contract,
    predictor_contracts,
)
from drevalpy.components.feature_block import BlockSpec
from drevalpy.components.predictors.block import BlockPredictor
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.matrix import MatrixPredictor
from drevalpy.components.registry import (
    get_cell_line_featurizer,
    get_drug_featurizer,
    get_predictor,
)

if TYPE_CHECKING:
    from drevalpy.models.config.featurizer import FeaturizerConfig
    from drevalpy.models.config.model import ModelConfig


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


def _featurizer_contract(cls: type[Any]):
    try:
        return featurizer_contract(cls)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def _predictor_contracts(cls: type[Any]):
    try:
        return predictor_contracts(cls)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def _is_feature_free(pred_cls: type[Any]) -> bool:
    return pred_cls.input_interface == "feature_free"


def _allows_no_featurizers(pred_cls: type[Any]) -> bool:
    return _is_feature_free(pred_cls)


def _validate_scope(config: ModelConfig, pred_cls: type[Any]) -> None:
    from drevalpy.types.model_scope import ModelScope

    supported_scopes = getattr(pred_cls, "supported_scopes", None)
    if supported_scopes is not None and config.scope not in supported_scopes:
        msg = (
            f"Predictor {config.predictor.name!r} does not support "
            f"scope={config.scope!r}; supported_scopes={sorted(supported_scopes)}"
        )
        raise ValueError(msg)
    if config.scope != ModelScope.SINGLE_DRUG or _allows_no_featurizers(pred_cls):
        return
    routing_featurizer = getattr(pred_cls, "routing_drug_featurizer", None)
    if routing_featurizer != "identity":
        msg = (
            f"Feature-based single-drug predictor {config.predictor.name!r} must declare "
            "routing_drug_featurizer='identity'"
        )
        raise ValueError(msg)
    if config.drug_featurizer is None or config.drug_featurizer.name != routing_featurizer:
        msg = (
            f"Feature-based single-drug predictor {config.predictor.name!r} requires "
            "drug_featurizer='identity' for per-drug routing"
        )
        raise ValueError(msg)


def _validate_prediction_mode(config: ModelConfig, pred_cls: type[Any]) -> None:
    supported = getattr(pred_cls, "supported_modes", None)
    if supported is not None and config.prediction_mode not in supported:
        msg = (
            f"Predictor {config.predictor.name!r} does not support "
            f"prediction_mode={config.prediction_mode!r}; "
            f"supported_modes={sorted(supported)}"
        )
        raise ValueError(msg)


def _validate_no_featurizer_predictor(config: ModelConfig, pred_cls: type[Any]) -> bool:
    """Return True when validation is complete for a feature-free predictor.

    :param config: Model configuration to validate.
    :param pred_cls: Resolved predictor class.
    :returns: ``True`` when the config is a valid feature-free predictor setup.
    :raises ValueError: If featurizers are incompatible with the predictor type.
    """
    if config.cell_line_featurizer is not None or config.drug_featurizer is not None:
        if _allows_no_featurizers(pred_cls):
            msg = f"Predictor {config.predictor.name!r} is a feature-free predictor and forbids configured featurizers"
            raise ValueError(msg)
        return False
    if not _allows_no_featurizers(pred_cls):
        msg = (
            f"Predictor {config.predictor.name!r} requires featurizers; "
            "set cell_line_featurizer and drug_featurizer, or use a feature-free predictor."
        )
        raise ValueError(msg)
    _validate_prediction_mode(config, pred_cls)
    return True


def _validate_featurizer_presence(config: ModelConfig, pred_cls: type[Any]) -> None:
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
        msg = f"drug_featurizer must use registry='drug', got {config.drug_featurizer.registry!r}"
        raise ValueError(msg)


def _validate_featurizer_views(config: ModelConfig) -> None:
    if config.cell_line_featurizer is not None:
        _validate_view_fields(config.cell_line_featurizer, label="cell_line_featurizer")
    if config.drug_featurizer is not None:
        _validate_view_fields(config.drug_featurizer, label="drug_featurizer")


def _validate_matrix_formats(config: ModelConfig, pred_cls: type[Any]) -> None:
    if pred_cls.input_interface != "matrix":
        return
    if config.cell_line_featurizer is not None:
        cell_cls = get_cell_line_featurizer(config.cell_line_featurizer.name)
        cell_contract = _featurizer_contract(cell_cls)
        if cell_contract.format != FeatureFormat.NUMERIC_MATRIX:
            msg = (
                f"Matrix predictor {config.predictor.name!r} requires numeric_matrix "
                f"cell-line features, got {cell_contract.format.value!r}"
            )
            raise ValueError(msg)
    if config.drug_featurizer is not None:
        drug_cls = get_drug_featurizer(config.drug_featurizer.name)
        drug_contract = _featurizer_contract(drug_cls)
        if drug_contract.format != FeatureFormat.NUMERIC_MATRIX:
            msg = (
                f"Matrix predictor {config.predictor.name!r} requires numeric_matrix "
                f"drug features, got {drug_contract.format.value!r}"
            )
            raise ValueError(msg)


def _validate_featurizer_contracts(config: ModelConfig, pred_cls: type[Any]) -> None:
    required_cell_line, required_drug = _predictor_contracts(pred_cls)
    if config.cell_line_featurizer is not None:
        cell_line_cls = get_cell_line_featurizer(config.cell_line_featurizer.name)
        cell_line_contract = _featurizer_contract(cell_line_cls)
        if not contracts_compatible(cell_line_contract, required_cell_line):
            msg = (
                f"Cell line featurizer contract {cell_line_contract!r} is incompatible with "
                f"predictor cell_line_contract {required_cell_line!r}"
            )
            raise ValueError(msg)
    if config.drug_featurizer is not None:
        drug_cls = get_drug_featurizer(config.drug_featurizer.name)
        drug_contract = _featurizer_contract(drug_cls)
        if not contracts_compatible(drug_contract, required_drug):
            msg = (
                f"Drug featurizer contract {drug_contract!r} is incompatible "
                f"with predictor drug_contract {required_drug!r}"
            )
            raise ValueError(msg)
    _validate_matrix_formats(config, pred_cls)


def _concat_child_block_specs(config: FeaturizerConfig) -> tuple[BlockSpec, ...]:
    from drevalpy.models.config.featurizer import FeaturizerConfig as FeaturizerConfigModel

    specs: list[BlockSpec] = []
    for child in config.hyperparameters.get("featurizers", []):
        child_config = (
            child if isinstance(child, FeaturizerConfigModel) else FeaturizerConfigModel.model_validate(child)
        )
        specs.extend(_block_specs_for_featurizer(child_config))
    return tuple(specs)


def _declared_or_view_block_specs(config: FeaturizerConfig) -> tuple[BlockSpec, ...]:
    cls = get_cell_line_featurizer(config.name) if config.registry == "cell_line" else get_drug_featurizer(config.name)
    declared = getattr(cls, "output_block_specs", ())
    if declared:
        return tuple(spec for spec in declared if isinstance(spec, BlockSpec))
    view = config.view or config.hyperparameters.get("view") or getattr(cls, "_default_view", None)
    if isinstance(view, str):
        return (BlockSpec(view, _featurizer_contract(cls).format),)
    return ()


def _block_specs_for_featurizer(config: FeaturizerConfig) -> tuple[BlockSpec, ...]:
    """Resolve the named blocks emitted by a configured featurizer tree.

    :param config: Featurizer config node to inspect.
    :returns: Block specs emitted by the featurizer tree.
    """
    if config.name == "concatFeaturizers":
        return _concat_child_block_specs(config)
    if config.name == "sparsegoOntology":
        input_type = str(config.hyperparameters.get("input_type", "expression"))
        name = "mutations" if input_type == "mutations" else "gene_expression"
        return (BlockSpec(name, FeatureFormat.NUMERIC_MATRIX, metadata=True),)
    return _declared_or_view_block_specs(config)


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
        emitted = _block_specs_for_featurizer(featurizer)
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


def _validate_leaf_interface(pred_cls: type[Any], predictor_name: str) -> None:
    leaf_bases = (FeatureFreePredictor, MatrixPredictor, BlockPredictor)
    matches = [base for base in leaf_bases if issubclass(pred_cls, base)]
    if len(matches) != 1:
        msg = (
            f"Predictor {predictor_name!r} must inherit exactly one of "
            f"FeatureFreePredictor, MatrixPredictor, BlockPredictor; "
            f"matched={[base.__name__ for base in matches]}"
        )
        raise ValueError(msg)


def validate_model_config(config: ModelConfig) -> None:
    """Check registry slots, feature compatibility, and prediction mode.

    :param config: Model configuration to validate.
    """
    pred_cls = get_predictor(config.predictor.name)
    _validate_leaf_interface(pred_cls, config.predictor.name)
    _validate_scope(config, pred_cls)
    if _validate_no_featurizer_predictor(config, pred_cls):
        return
    _validate_featurizer_presence(config, pred_cls)
    _validate_featurizer_views(config)
    _validate_prediction_mode(config, pred_cls)
    _validate_featurizer_contracts(config, pred_cls)
    _validate_block_schema(config, pred_cls)
