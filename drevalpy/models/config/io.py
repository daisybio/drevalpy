"""Parse declarative model configs from strings, dicts, and YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from drevalpy.models.config._from_dict import from_dict
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.models.config.spec import (
    apply_optional_hyperparameters,
    recipe_payload,
    zoo_config,
)
from drevalpy.types.prediction_mode import PredictionMode

__all__ = ["from_dict", "from_spec", "from_yaml"]


def from_spec(
    spec: str,
    *,
    hyperparameters: dict[str, Any] | None = None,
    prediction_mode: PredictionMode | str | None = None,
) -> ModelConfig | ResolvedModelConfig:
    """Build a ``ModelConfig`` from a zoo preset name or a recipe string.

    A spec is either the name of a registered zoo preset or a recipe naming the parts
    directly. Zoo names win, so a preset can shadow a bare predictor name. Recipes take the
    same two steps as any other config source: ``recipe_payload`` reads the syntax into a
    field mapping, then ``from_dict`` resolves the names against the registry and checks
    that the combination is legal.

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

    config = from_dict(recipe_payload(trimmed, prediction_mode=mode), source=f"recipe {trimmed!r}")
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
