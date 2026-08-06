"""Parse declarative model configs from strings, dicts, and YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from drevalpy.models.config.model import ModelConfig
from drevalpy.models.config.spec import build_model_config_from_spec
from drevalpy.types.prediction_mode import PredictionMode


def _format_validation_error(exc: ValidationError, *, source: Path | str | None = None) -> str:
    prefix = "Invalid model config"
    if source is not None:
        prefix = f"{prefix} in {source}"
    details = "; ".join(f"{' -> '.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())
    return f"{prefix}: {details}"


def model_config_from_dict(data: dict[str, Any], *, source: Path | str | None = None) -> ModelConfig:
    """Build a ``ModelConfig`` from a plain dictionary.

    :param data: Mapping with featurizer and predictor sections.
    :param source: Optional path or label included in validation error messages.
    :returns: Validated ``ModelConfig`` instance.
    :raises ValueError: If validation fails.
    """
    try:
        return ModelConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(_format_validation_error(exc, source=source)) from exc


def model_config_from_spec(
    spec: str,
    *,
    hyperparameters: dict[str, Any] | None = None,
    prediction_mode: str | None = None,
) -> ModelConfig:
    """Build a ``ModelConfig`` from a recipe, zoo, legacy, or baseline spec.

    :param spec: Zoo preset name, colon-separated recipe, or legacy baseline token.
    :param hyperparameters: Optional flat public hyperparameter overrides.
    :param prediction_mode: Optional prediction mode string; defaults to regression.
    :returns: Validated ``ModelConfig`` instance.
    """
    if prediction_mode is None:
        return build_model_config_from_spec(spec, hyperparameters=hyperparameters)
    return build_model_config_from_spec(
        spec,
        hyperparameters=hyperparameters,
        prediction_mode=PredictionMode(prediction_mode),
    )


def model_config_from_yaml(path: Path | str) -> ModelConfig:
    """Load a ``ModelConfig`` from a YAML file.

    :param path: Path to a YAML mapping describing the model config.
    :returns: Validated ``ModelConfig`` instance.
    :raises FileNotFoundError: If ``path`` does not exist.
    :raises ValueError: If the YAML content is not a valid config mapping.
    """
    yaml_path = Path(path)
    if not yaml_path.is_file():
        msg = f"Model config YAML not found: {yaml_path}"
        raise FileNotFoundError(msg)
    with yaml_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"Model config YAML must contain a mapping: {yaml_path}"
        raise ValueError(msg)
    return model_config_from_dict(data, source=yaml_path)
