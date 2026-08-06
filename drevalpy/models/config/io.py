"""Parse declarative model configs from strings, dicts, and YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from drevalpy.models.config._from_dict import from_dict
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.models.config.spec import _build_from_spec
from drevalpy.types.prediction_mode import PredictionMode

__all__ = ["from_dict", "from_spec", "from_yaml"]


def from_spec(
    spec: str,
    *,
    hyperparameters: dict[str, Any] | None = None,
    prediction_mode: str | None = None,
) -> ModelConfig | ResolvedModelConfig:
    """Build a ``ModelConfig`` from a zoo name or a recipe string.

    :param spec: Zoo preset name, or a colon-separated recipe.
    :param hyperparameters: Optional flat public hyperparameter overrides.
    :param prediction_mode: Optional prediction mode string; defaults to regression.
    :returns: Validated ``ModelConfig`` template, or ``ResolvedModelConfig`` when
        *hyperparameters* are provided.
    """
    if prediction_mode is None:
        return _build_from_spec(spec, hyperparameters=hyperparameters)
    return _build_from_spec(
        spec,
        hyperparameters=hyperparameters,
        prediction_mode=PredictionMode(prediction_mode),
    )


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
