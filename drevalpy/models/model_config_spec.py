"""Build `~drevalpy.models.config.ModelConfig` from recipe, zoo, or legacy names."""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.model_id import parse_model_id
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.raw_dataset import RawDatasetPredictor
from drevalpy.models.config import (
    FeaturizerConfig,
    ModelConfig,
    ModelScope,
    PredictionMode,
    PredictorConfig,
)


def _coerce_prediction_mode(mode: PredictionMode | str) -> PredictionMode:
    if isinstance(mode, PredictionMode):
        return mode
    return PredictionMode(mode)


def _default_scope_for_predictor(pred_cls: type[Any]) -> ModelScope:
    supported_scopes = getattr(pred_cls, "supported_scopes", None)
    if supported_scopes is not None and len(supported_scopes) == 1:
        return next(iter(supported_scopes))
    return ModelScope.MULTI_DRUG


def _config_from_recipe_triple(
    spec: str,
    *,
    hyperparameters: dict[str, Any] | None = None,
    prediction_mode: PredictionMode | str = PredictionMode.REGRESSION,
) -> ModelConfig:
    from drevalpy.components.registry import get_predictor

    cell_line_type, drug_type, predictor_type = parse_model_id(spec.strip())
    pred_cls = get_predictor(predictor_type)
    predictor = PredictorConfig(name=predictor_type, hyperparameters=dict(hyperparameters or {}))
    mode = _coerce_prediction_mode(prediction_mode)
    scope = _default_scope_for_predictor(pred_cls)
    if cell_line_type is None:
        config = ModelConfig(
            cell_line_featurizer=None,
            drug_featurizer=None,
            predictor=predictor,
            prediction_mode=mode,
            scope=scope,
        )
        config.validate()
        return config
    if drug_type is None:
        msg = "recipe triple requires a drug featurizer when a cell-line featurizer is set"
        raise ValueError(msg)
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig.model_validate(
            normalize_featurizer_config(cell_line_type, default_registry="cell_line"),
        ),
        drug_featurizer=FeaturizerConfig.model_validate(
            normalize_featurizer_config(drug_type, default_registry="drug"),
        ),
        predictor=predictor,
        prediction_mode=mode,
        scope=scope,
    )
    config.validate()
    return config


def _config_from_no_featurizer_predictor_token(
    token: str,
    *,
    prediction_mode: PredictionMode | str = PredictionMode.REGRESSION,
) -> ModelConfig | None:
    from drevalpy.components.registry import get_predictor

    try:
        pred_cls = get_predictor(token)
    except (ValueError, ImportError):
        return None
    if not (issubclass(pred_cls, FeatureFreePredictor) or issubclass(pred_cls, RawDatasetPredictor)):
        return None
    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name=token),
        prediction_mode=_coerce_prediction_mode(prediction_mode),
    )
    # Preserve predictor-declared default scope for single-drug raw models.
    config = config.model_copy(update={"scope": _default_scope_for_predictor(pred_cls)}, deep=True)
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
    3. Zoo / factory model name (PascalCase)
    4. Feature-free or raw predictor token (no featurizers required), e.g. ``naiveMean`` or ``dipk``
    """
    from drevalpy.models.factory import model_config_for_name

    trimmed = spec.strip()
    if not trimmed:
        msg = "model spec must be a non-empty string"
        raise ValueError(msg)

    mode = _coerce_prediction_mode(prediction_mode)

    if ":" in trimmed:
        return _config_from_recipe_triple(
            trimmed,
            hyperparameters=hyperparameters,
            prediction_mode=mode,
        )

    try:
        config = model_config_for_name(trimmed, hyperparameters)
    except KeyError:
        config = None
    if config is not None:
        if config.prediction_mode != mode:
            config = config.model_copy(update={"prediction_mode": mode}, deep=True)
            config.validate()
        return config

    no_feat = _config_from_no_featurizer_predictor_token(trimmed, prediction_mode=prediction_mode)
    if no_feat is not None:
        if hyperparameters:
            return no_feat.model_copy(
                update={
                    "predictor": no_feat.predictor.model_copy(
                        update={"hyperparameters": {**no_feat.predictor.hyperparameters, **hyperparameters}}
                    )
                },
                deep=True,
            )
        return no_feat

    msg = (
        f"Unknown model spec {spec!r}. Use a recipe triple "
        "(cellLine:drug:predictor), zoo name, or feature-free/raw predictor token."
    )
    raise ValueError(msg)
