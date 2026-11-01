"""Parse declarative model configs from strings, dicts, and YAML files."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import ValidationError
from upath import UPath as Path

from drevalpy.models.config._parse import parse_model_recipe
from drevalpy.models.config.model import ModelConfig
from drevalpy.registry.predictor import predictor_registry

__all__ = ["from_dict", "from_recipe", "from_yaml"]


def _details(exc: ValidationError) -> str:
    """Join the errors of *exc* as ``location: message``, without pydantic's ``Value error, `` prefix."""
    entries = []
    for error in exc.errors():
        message = str(error["ctx"]["error"]) if error["type"] == "value_error" else str(error["msg"])
        location = " -> ".join(str(part) for part in error["loc"])
        entries.append(f"{location}: {message}" if location else message)
    return "; ".join(entries)


def from_dict(data: dict[str, Any], *, source: Path | str | None = None) -> ModelConfig:
    """Build a ``ModelConfig`` from a plain dictionary.

    This is where the registry is consulted: field validation resolves featurizer and
    predictor names, and the model-level validator checks that the combination is legal.

    :param data: Mapping with featurizer and predictor sections.
    :param source: Optional path or label included in validation error messages.
    :returns: Validated ``ModelConfig`` instance.
    :raises ValueError: If validation fails.
    """
    try:
        return ModelConfig.model_validate(data)
    except ValidationError as exc:
        where = "" if source is None else f" in {source}"
        raise ValueError(f"Invalid model config{where}: {_details(exc)}") from exc


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

    ``parse_model_recipe`` reads the syntax into the plain field mapping a YAML file would hold,
    then validation resolves the names against the registry and checks that the combination is
    legal. The error says why the recipe is invalid, without repeating the recipe. Zoo names are
    resolved by :meth:`drevalpy.models.spec.ModelSpec.from_name`, not here.

    :param recipe: A recipe of one to three colon-separated parts, such as
        ``"pca[gene_expression]:fingerprints:elasticNet"``.
    :returns: Validated ``ModelConfig``.
    :raises ValueError: If *recipe* is empty, malformed, names an unknown predictor, or fails validation.
    """
    payload = parse_model_recipe(recipe)
    if payload["sample_featurizer"] is None:
        _reject_unknown_token(payload["predictor"])
    try:
        return ModelConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(_details(exc)) from exc


def read_yaml_mapping(path: Path | str, what: str) -> dict[str, Any]:
    """Read a YAML file whose top-level node is a mapping.

    :param path: YAML file.
    :param what: What the file holds, for the messages (``"Model config YAML"``).
    :returns: The parsed mapping.
    :raises FileNotFoundError: If ``path`` does not exist.
    :raises TypeError: If the YAML top-level node is not a mapping.
    """
    yaml_path = Path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"{what} not found: {yaml_path}")
    with yaml_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{what} must contain a mapping: {yaml_path}")
    return data


def from_yaml(path: Path | str) -> ModelConfig:
    """Load a ``ModelConfig`` from a YAML file.

    ``read_yaml_mapping`` raises ``FileNotFoundError`` for a missing file and ``TypeError`` when
    the top-level node is not a mapping.

    :param path: Path to a YAML mapping describing the model config.
    :returns: Validated ``ModelConfig`` instance.
    """
    return from_dict(read_yaml_mapping(path, "Model config YAML"), source=Path(path))
