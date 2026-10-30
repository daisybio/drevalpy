"""Parse declarative model configs from strings, dicts, and YAML files."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import ValidationError
from upath import UPath as Path

from drevalpy.models.config._recipe import parse_model_recipe
from drevalpy.models.config.model import ModelConfig
from drevalpy.registry.predictor import predictor_registry

__all__ = ["from_dict", "from_recipe", "from_yaml"]


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


def _reject_unknown_token(token: str) -> None:
    """Report a bare token that names no predictor drevalpy knows.

    A bare recipe names only a predictor, so an unknown one is most likely a mistyped zoo name
    and is reported as such rather than as a predictor-shaped config error.

    :param token: The single-part recipe.
    :raises ValueError: If *token* names no registered predictor.
    """
    if token in predictor_registry:
        return
    raise ValueError(
        f"Unknown model spec {token!r}. Use a recipe triple (sample:drug:predictor), zoo name, or feature-free "
        "predictor token."
    )


def from_recipe(recipe: str) -> ModelConfig:
    """Build a ``ModelConfig`` from a recipe string.

    A recipe takes the same two steps as any other config source: ``parse_model_recipe`` reads
    the syntax into a plain field mapping, then ``from_dict`` resolves the names against the
    registry and checks that the combination is legal. Zoo names are resolved by
    :meth:`drevalpy.models.spec.ModelSpec.from_name`, not here.

    :param recipe: A recipe of one to three colon-separated parts, such as
        ``"pca[gene_expression]:fingerprints:elasticNet"``.
    :returns: Validated ``ModelConfig``.
    :raises ValueError: If *recipe* is empty, malformed, names an unknown predictor, or fails validation.
    """
    trimmed = recipe.strip()
    if not trimmed:
        raise ValueError("model spec must be a non-empty string")
    payload = parse_model_recipe(trimmed)
    if payload["sample_featurizer"] is None:
        _reject_unknown_token(payload["predictor"])
    return from_dict(payload, source=f"recipe {trimmed!r}")


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
