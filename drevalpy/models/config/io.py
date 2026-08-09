"""Parse declarative model configs from strings, dicts, and YAML files."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import ValidationError
from upath import UPath as Path

from drevalpy.models.config._recipe import parse_model_recipe
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.models.config.spec import (
    apply_optional_hyperparameters,
    reject_unknown_spec,
    zoo_config,
)
from drevalpy.types.prediction_mode import PredictionMode

__all__ = ["from_dict", "from_spec", "from_yaml"]


def _format_error_entry(error: Any) -> str:
    location = " -> ".join(str(part) for part in error["loc"])
    if not location:
        return str(error["msg"])
    return f"{location}: {error['msg']}"


def _format_validation_error(exc: ValidationError, *, source: Path | str | None = None) -> str:
    prefix = "Invalid model config"
    if source is not None:
        prefix = f"{prefix} in {source}"
    details = "; ".join(_format_error_entry(error) for error in exc.errors())
    return f"{prefix}: {details}"


def from_dict(data: dict[str, Any], *, source: Path | str | None = None) -> ModelConfig:
    """Build a ``ModelConfig`` from a plain dictionary.

    This is where the registry is consulted: field validation resolves featurizer and
    predictor names, and the model-level validator checks that the combination is legal.
    Every other entry point in this module reduces its source to a mapping and ends up here.

    :param data: Mapping with featurizer and predictor sections.
    :param source: Optional path or label included in validation error messages.
    :returns: Validated ``ModelConfig`` instance.
    :raises ValueError: If validation fails.
    """
    try:
        return ModelConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(_format_validation_error(exc, source=source)) from exc


def from_spec(
    spec: str,
    *,
    hyperparameters: dict[str, Any] | None = None,
    prediction_mode: PredictionMode | str | None = None,
) -> ModelConfig | ResolvedModelConfig:
    """Build a ``ModelConfig`` from a zoo preset name or a recipe string.

    A spec is either the name of a registered zoo preset or a recipe naming the parts
    directly. Zoo names win, so a preset can shadow a bare predictor name. Recipes take the
    same two steps as any other config source: ``parse_model_recipe`` reads the syntax into a
    plain field mapping, then ``from_dict`` resolves the names against the registry and checks
    that the combination is legal. A bare token is exactly the recipe with no cell-line slot,
    so that is where a mistyped zoo name is caught before it is read as a predictor.

    :param spec: Zoo preset name, or a recipe of one to three colon-separated parts.
    :param hyperparameters: Optional flat public hyperparameter overrides.
    :param prediction_mode: Prediction mode for the predictor; defaults to regression.
    :returns: Validated ``ModelConfig`` template, or ``ResolvedModelConfig`` when
        *hyperparameters* are provided.
    :raises ValueError: If *spec* is unknown or validation fails.
    """
    trimmed = spec.strip()
    if not trimmed:
        msg = "model spec must be a non-empty string"
        raise ValueError(msg)
    mode = PredictionMode.REGRESSION if prediction_mode is None else PredictionMode(prediction_mode)

    preset = zoo_config(trimmed, hyperparameters, mode)
    if preset is not None:
        return preset

    payload = parse_model_recipe(trimmed)
    if payload["cell_line_featurizer"] is None:
        reject_unknown_spec(payload["predictor"])
    config = from_dict({**payload, "prediction_mode": mode}, source=f"recipe {trimmed!r}")
    return apply_optional_hyperparameters(config, hyperparameters)


def from_yaml(path: Path | str) -> ModelConfig:
    """Load a ``ModelConfig`` from a YAML file.

    :param path: Path to a YAML mapping describing the model config.
    :returns: Validated ``ModelConfig`` instance.
    :raises FileNotFoundError: If ``path`` does not exist.
    :raises TypeError: If the YAML top-level node is not a mapping.
    """
    yaml_path = Path(path)
    if not yaml_path.is_file():
        msg = f"Model config YAML not found: {yaml_path}"
        raise FileNotFoundError(msg)
    with yaml_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"Model config YAML must contain a mapping: {yaml_path}"
        raise TypeError(msg)
    return from_dict(data, source=yaml_path)
