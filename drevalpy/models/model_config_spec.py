"""Build `~drevalpy.models.config.ModelConfig` from recipe, zoo, or legacy names."""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.model_id import parse_model_id
from drevalpy.components.predictors.baseline import BaselinePredictor
from drevalpy.models.config import (
    FeaturizerConfig,
    ModelConfig,
    PredictionMode,
    PredictorConfig,
)


def _coerce_prediction_mode(mode: PredictionMode | str) -> PredictionMode:
    if isinstance(mode, PredictionMode):
        return mode
    return PredictionMode(mode)


def _config_from_recipe_triple(
    spec: str,
    *,
    hyperparameters: dict[str, Any] | None = None,
) -> ModelConfig:
    cell_line_type, drug_type, predictor_type = parse_model_id(spec.strip())
    predictor = PredictorConfig(name=predictor_type, hyperparameters=dict(hyperparameters or {}))
    if cell_line_type is None:
        return ModelConfig(
            cell_line_featurizer=None,
            drug_featurizer=None,
            predictor=predictor,
        )
    if drug_type is None:
        msg = "recipe triple requires a drug featurizer when a cell-line featurizer is set"
        raise ValueError(msg)
    return ModelConfig(
        cell_line_featurizer=FeaturizerConfig.model_validate(
            normalize_featurizer_config(cell_line_type, default_registry="cell_line"),
        ),
        drug_featurizer=FeaturizerConfig.model_validate(
            normalize_featurizer_config(drug_type, default_registry="drug"),
        ),
        predictor=predictor,
    )


def _config_from_baseline_predictor_token(
    token: str,
    *,
    prediction_mode: PredictionMode | str = PredictionMode.REGRESSION,
) -> ModelConfig | None:
    from drevalpy.components.registry import get_predictor

    try:
        pred_cls = get_predictor(token)
    except ValueError:
        return None
    if not (issubclass(pred_cls, BaselinePredictor) or getattr(pred_cls, "category", "") == "baseline"):
        return None
    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name=token),
        prediction_mode=_coerce_prediction_mode(prediction_mode),
    )
    config.validate()
    return config


def build_model_config_from_spec(
    spec: str,
    *,
    hyperparameters: dict[str, Any] | None = None,
    prediction_mode: PredictionMode | str = PredictionMode.REGRESSION,
) -> ModelConfig:
    """Parse a model specification string.

    Resolution order:

    1. ``cellLine:drug:predictor`` registry triple
    2. Built-in or external zoo preset name
    3. Legacy ``MODEL_FACTORY`` model name (PascalCase)
    4. Baseline predictor token (no featurizers required), e.g. ``naiveMean`` or ``dipk``
    """
    from drevalpy.components.register_builtins import ensure_components_registered
    from drevalpy.models.factory import model_config_for_name

    trimmed = spec.strip()
    if not trimmed:
        msg = "model spec must be a non-empty string"
        raise ValueError(msg)

    ensure_components_registered()

    if ":" in trimmed:
        return _config_from_recipe_triple(trimmed, hyperparameters=hyperparameters)

    try:
        return model_config_for_name(trimmed, hyperparameters)
    except KeyError:
        pass

    baseline = _config_from_baseline_predictor_token(trimmed, prediction_mode=prediction_mode)
    if baseline is not None:
        if hyperparameters:
            return baseline.model_copy(
                update={
                    "predictor": baseline.predictor.model_copy(
                        update={"hyperparameters": {**baseline.predictor.hyperparameters, **hyperparameters}}
                    )
                },
                deep=True,
            )
        return baseline

    msg = (
        f"Unknown model spec {spec!r}. Use a recipe triple "
        "(cellLine:drug:predictor), zoo name, legacy model name, or baseline predictor token."
    )
    raise ValueError(msg)
