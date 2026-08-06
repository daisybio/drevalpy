"""Build `~drevalpy.models.config.ModelConfig` from recipe, zoo, or legacy names."""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.model_id import parse_model_id
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.models.config.featurizer import CellLineFeaturizerConfig, DrugFeaturizerConfig
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.config.predictor import PredictorConfig
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.types.model_scope import ModelScope
from drevalpy.types.prediction_mode import PredictionMode


def _coerce_prediction_mode(mode: PredictionMode | str) -> PredictionMode:
    if isinstance(mode, PredictionMode):
        return mode
    return PredictionMode(mode)


def _default_scope_for_predictor(pred_cls: type[Any]) -> ModelScope:
    supported_scopes = getattr(pred_cls, "supported_scopes", None)
    if supported_scopes is not None and len(supported_scopes) == 1:
        return next(iter(supported_scopes))
    return ModelScope.MULTI_DRUG


def _apply_optional_hyperparameters(
    config: ModelConfig,
    hyperparameters: dict[str, Any] | None,
) -> ModelConfig | ResolvedModelConfig:
    """Apply public hyperparameters by returning a resolved config when needed.

    Recipe builders historically returned ``ModelConfig``. When hyperparameters
    are provided, return the resolved object so callers that only need a template
    without overrides still receive ``ModelConfig``, while override paths receive
    ``ResolvedModelConfig`` via the public-flat helper.

    :param config: Template config.
    :param hyperparameters: Optional public overrides.
    :returns: Template or resolved config.
    """
    if not hyperparameters:
        return config
    from drevalpy.components.tuning.public_flat import apply_public_hyperparameters_to_config

    return apply_public_hyperparameters_to_config(config, hyperparameters)


def _config_from_recipe_triple(
    spec: str,
    *,
    hyperparameters: dict[str, Any] | None = None,
    prediction_mode: PredictionMode | str = PredictionMode.REGRESSION,
) -> ModelConfig | ResolvedModelConfig:
    from drevalpy.components.registry import get_predictor

    cell_line_type, drug_type, predictor_type = parse_model_id(spec.strip())
    pred_cls = get_predictor(predictor_type)
    predictor = PredictorConfig(name=predictor_type)
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
        return _apply_optional_hyperparameters(config, hyperparameters)
    if drug_type is None:
        if scope != ModelScope.SINGLE_DRUG:
            msg = "two-part recipes require a single-drug predictor"
            raise ValueError(msg)
        config = ModelConfig(
            cell_line_featurizer=CellLineFeaturizerConfig.model_validate(
                normalize_featurizer_config(cell_line_type, default_registry="cell_line"),
            ),
            drug_featurizer=None,
            predictor=predictor,
            prediction_mode=mode,
            scope=scope,
        )
        return _apply_optional_hyperparameters(config, hyperparameters)
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig.model_validate(
            normalize_featurizer_config(cell_line_type, default_registry="cell_line"),
        ),
        drug_featurizer=DrugFeaturizerConfig.model_validate(
            normalize_featurizer_config(drug_type, default_registry="drug"),
        ),
        predictor=predictor,
        prediction_mode=mode,
        scope=scope,
    )
    return _apply_optional_hyperparameters(config, hyperparameters)


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
    if not issubclass(pred_cls, FeatureFreePredictor):
        return None
    return ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name=token),
        prediction_mode=_coerce_prediction_mode(prediction_mode),
        scope=_default_scope_for_predictor(pred_cls),
    )


def _build_from_spec(
    spec: str,
    *,
    hyperparameters: dict[str, Any] | None = None,
    prediction_mode: PredictionMode | str = PredictionMode.REGRESSION,
) -> ModelConfig | ResolvedModelConfig:
    """Parse a model specification string.

    Resolution order:

    1. ``cellLine:drug:predictor`` registry triple
    2. Built-in or external zoo preset name
    3. Zoo / factory model name (PascalCase)
    4. Feature-free predictor token (no featurizers required), e.g. ``naiveMean``

    :param spec: Zoo preset name, recipe triple, or feature-free predictor token.
    :param hyperparameters: Optional flat public hyperparameter overrides.
    :param prediction_mode: Regression or classification mode for the predictor.
    :returns: Validated ``ModelConfig`` instance, or ``ResolvedModelConfig`` when
        *hyperparameters* are provided.
    :raises ValueError: If ``spec`` is unknown or validation fails.
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
        # model_config_for_name already applied hyperparameters when provided.
        if hyperparameters:
            return config
        if isinstance(config, ResolvedModelConfig):
            return config
        if config.prediction_mode != mode:
            return config.replace(prediction_mode=mode)
        return config

    no_feat = _config_from_no_featurizer_predictor_token(trimmed, prediction_mode=prediction_mode)
    if no_feat is not None:
        return _apply_optional_hyperparameters(no_feat, hyperparameters)

    msg = (
        f"Unknown model spec {spec!r}. Use a recipe triple "
        "(cellLine:drug:predictor), zoo name, or feature-free predictor token."
    )
    raise ValueError(msg)
