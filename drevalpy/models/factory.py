"""Resolve zoo/spec names to `~drevalpy.models.config.ModelConfig` objects.

Also contains the typo guard and hyperparameter-application helper that
``drevalpy.models.config.io.from_spec`` composes around recipe parsing.
"""

from __future__ import annotations

from typing import Any

from drevalpy.models.config.model import ModelConfig
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.registry._builtins import is_known_builtin_predictor
from drevalpy.registry.predictor import list as list_predictors
from drevalpy.types.enums.prediction_mode import PredictionMode


def model_config_for_name(
    model_name: str,
    hyperparameters: dict[str, Any] | None = None,
    *,
    prediction_mode: PredictionMode | None = None,
) -> ModelConfig | ResolvedModelConfig:
    """Resolve a factory/zoo name to a modular config with public flat HP applied.

    :param model_name: Built-in or external zoo preset name.
    :param hyperparameters: Optional flat public hyperparameter overrides.
    :param prediction_mode: Optional prediction mode overriding the preset's own value.
    :returns: Template ``ModelConfig``, or ``ResolvedModelConfig`` when overrides are given.
    :raises KeyError: If ``model_name`` is not a known zoo entry.
    """
    from drevalpy.models.zoo import list_zoo_names, zoo_model_config

    if model_name not in list_zoo_names(include_external=True):
        msg = f"Unknown model name: {model_name}"
        raise KeyError(msg)
    return zoo_model_config(model_name, hyperparameters, prediction_mode=prediction_mode)


def apply_optional_hyperparameters(
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
    from drevalpy.models.tuning.public_flat import apply_public_hyperparameters_to_config

    return apply_public_hyperparameters_to_config(config, hyperparameters)


def reject_unknown_spec(token: str) -> None:
    """Report a bare token that names neither a zoo preset nor a predictor drevalpy knows.

    Such a token is most likely a mistyped zoo name, so it is reported in terms of both
    options rather than as a predictor-shaped config error. A name in the built-in catalog or
    in the registry is passed through instead, so that ``from_dict`` reaches ``get_predictor``
    and the far more useful "is unavailable; its optional/literature dependency was not
    registered" message, or the underlying ``ImportError``, survives.

    The catalog is consulted first because ``list_predictors`` registers every built-in on the
    way, so a built-in token would otherwise be answered by whichever unrelated optional
    dependency happened to fail during that sweep.

    :param token: The single-part recipe, already known not to be a zoo preset.
    :raises ValueError: If *token* names no known built-in or registered predictor.
    """
    if is_known_builtin_predictor(token) or token in list_predictors():
        return
    msg = (
        f"Unknown model spec {token!r}. Use a recipe triple "
        "(cellLine:drug:predictor), zoo name, or feature-free predictor token."
    )
    raise ValueError(msg)


def zoo_config(
    name: str,
    hyperparameters: dict[str, Any] | None,
    prediction_mode: PredictionMode | str,
) -> ModelConfig | ResolvedModelConfig | None:
    """Resolve a registered zoo preset, or report that *name* is not one.

    :param name: Candidate zoo preset name.
    :param hyperparameters: Optional flat public hyperparameter overrides.
    :param prediction_mode: Mode to apply, honoured only when no overrides are given.
    :returns: The preset's config, or ``None`` when *name* is not a zoo entry.
    """
    try:
        return model_config_for_name(
            name,
            hyperparameters,
            prediction_mode=None if hyperparameters else PredictionMode(prediction_mode),
        )
    except KeyError:
        return None
