"""Validation logic for `~drevalpy.models.config.ModelConfig`.

A predictor declares what it reads from each side
(:func:`drevalpy.components.predictors.inputs.predictor_inputs`); a config is legal when each
side's featurizer is present exactly when that side reads something, and emits what it reads.
A single-drug predictor routes pairs by ``batch.drug_ids``, so it needs no drug featurizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._featurizer_tree import iter_featurizer_leaves
from drevalpy.components.predictors.inputs import Inputs, predictor_inputs
from drevalpy.models.config._block_specs import resolve_output_block_specs
from drevalpy.registry.featurizer import get_featurizer
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.feature_source import Side
from drevalpy.types.enums.model_scope import ModelScope

if TYPE_CHECKING:
    from drevalpy.models.config.featurizer import FeaturizerConfig
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


def _label(spec: BlockSpec) -> str:
    """Name a spec, adding its format when it is not a numeric matrix."""
    return spec.name if spec.format == FeatureFormat.NUMERIC_MATRIX else f"{spec.name}:{spec.format.value}"


def _describe(specs: tuple[BlockSpec, ...]) -> str:
    return ", ".join(_label(spec) for spec in specs)


def _validate_inputs(predictor: str, side: Side, inputs: Inputs, featurizer: FeaturizerConfig) -> None:
    """Check that *featurizer* emits what *inputs* reads.

    :param predictor: Predictor name, for the message.
    :param side: Side of *featurizer*.
    :param inputs: The predictor's declaration for *side*.
    :param featurizer: The side's featurizer config.
    :raises ValueError: If a required block is missing, or a matrix input gets a non-numeric block.
    """
    emitted = resolve_output_block_specs(featurizer)
    if inputs.kind == "matrix":
        if any(spec.format != FeatureFormat.NUMERIC_MATRIX for spec in emitted):
            raise ValueError(
                f"Predictor {predictor!r} reads one {FeatureFormat.NUMERIC_MATRIX.value} design matrix from the "
                f"{side} featurizer {featurizer.name!r}, which emits [{_describe(emitted)}]"
            )
        return
    provided = {(spec.name, spec.format) for spec in emitted}
    missing = [
        spec for spec in inputs.specs if isinstance(spec, BlockSpec) and (spec.name, spec.format) not in provided
    ]
    if missing:
        raise ValueError(
            f"Predictor {predictor!r} needs {side} block(s) {[_label(spec) for spec in missing]} "
            f"but the {side} featurizer {featurizer.name!r} emits [{_describe(emitted)}]"
        )
    for alternatives in inputs.specs:
        if isinstance(alternatives, tuple) and not provided.intersection((s.name, s.format) for s in alternatives):
            raise ValueError(
                f"Predictor {predictor!r} needs one of the {side} blocks [{_describe(alternatives)}] but the {side} "
                f"featurizer {featurizer.name!r} emits [{_describe(emitted)}]"
            )


def validate(config: ModelConfig) -> ModelConfig:
    """Check the prediction mode and that each side's featurizer matches the predictor's inputs.

    :param config: Model configuration to validate.
    :returns: The unchanged *config*, so this doubles as a Pydantic ``after`` validator.
    :raises ValueError: If a side has a featurizer it does not read, lacks one it reads, or the
        featurizer emits blocks the predictor cannot read.
    """
    name = config.predictor.name
    try:
        pred_cls = get_predictor(name)
    except ImportError:
        return config
    _validate_prediction_mode(config, pred_cls)
    sides: tuple[tuple[Side, FeaturizerConfig | None], ...] = (
        ("sample", config.sample_featurizer),
        ("drug", config.drug_featurizer),
    )
    for side, featurizer in sides:
        if featurizer is not None:
            for leaf in iter_featurizer_leaves(featurizer):
                get_featurizer(leaf.name, side)
        inputs = predictor_inputs(pred_cls, side)
        if inputs.kind == "none":
            if featurizer is not None:
                raise ValueError(f"Predictor {name!r} reads no {side} features and forbids a {side}_featurizer")
        elif featurizer is not None:
            _validate_inputs(name, side, inputs, featurizer)
        elif not (side == "drug" and pred_cls.scope is ModelScope.SINGLE_DRUG):
            raise ValueError(f"Predictor {name!r} requires a {side}_featurizer")
    return config
