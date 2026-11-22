"""Parse declarative model configs from strings, dicts, and YAML files."""

from __future__ import annotations

from typing import Any, get_args

import yaml
from pydantic import BaseModel, ValidationError
from upath import UPath as Path

from drevalpy.models.config._parse import DEFAULTS, parse_model_recipe
from drevalpy.models.config.featurizer import FeaturizerConfig
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.config.predictor import PredictorConfig
from drevalpy.registry.predictor import predictor_registry

__all__ = ["from_dict", "from_recipe", "from_yaml"]


_EXTRA_KEY_HINTS: dict[type[BaseModel], tuple[str, str]] = {
    FeaturizerConfig: ("featurizer", "arguments of the featurizer go under 'options'"),
    PredictorConfig: ("predictor", "hyperparameters go under 'defaults' or 'hyperparameter_space'"),
}
"""What a mistyped key of each config mapping is called, and where its value likely belongs."""


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    """Return the config model a field annotation holds, looking through ``| None`` and ``tuple[...]``.

    :param annotation: A resolved field annotation.
    :returns: The model class, or ``None`` when the field holds no model.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return next((model for arg in get_args(annotation) if (model := _nested_model(arg))), None)


def _owner(root: str, location: tuple[int | str, ...]) -> type[BaseModel] | None:
    """Return the config model whose mapping holds the key at *location*.

    :param root: Name of the model that was validated, the ``title`` of the error.
    :param location: Path of the key from that model, as pydantic reports it.
    :returns: The model owning the last part of *location*, or ``None`` when it is not a config model.
    """
    model = next((cls for cls in (ModelConfig, *_EXTRA_KEY_HINTS) if cls.__name__ == root), None)
    for part in location[:-1]:
        if model is None or isinstance(part, int):
            continue
        field = model.model_fields.get(part)
        model = None if field is None else _nested_model(field.annotation)
    return model


def _extra_key_hint(model: type[BaseModel] | None) -> str | None:
    """Say which keys a config mapping accepts and where a stray value likely belongs.

    :param model: The config model holding the unknown key.
    :returns: One sentence, or ``None`` for a model without a hint.
    """
    if model not in _EXTRA_KEY_HINTS:
        return None
    noun, where = _EXTRA_KEY_HINTS[model]
    keys = ", ".join([*model.model_fields, DEFAULTS])
    return f"A {noun} accepts the keys {keys}; {where}"


def _details(exc: ValidationError) -> str:
    """Join the errors of *exc* as ``location: message``, without pydantic's ``Value error, `` prefix.

    An unknown key of a featurizer or predictor mapping is followed, once, by the keys it accepts.

    :param exc: The validation error of a config model.
    :returns: The errors joined by ``; ``, then the hints as sentences.
    """
    entries: list[str] = []
    hints: list[str] = []
    for error in exc.errors():
        message = str(error["ctx"]["error"]) if error["type"] == "value_error" else str(error["msg"])
        location = " -> ".join(str(part) for part in error["loc"])
        entries.append(f"{location}: {message}" if location else message)
        hint = _extra_key_hint(_owner(exc.title, error["loc"])) if error["type"] == "extra_forbidden" else None
        if hint is not None and hint not in hints:
            hints.append(hint)
    return ". ".join(["; ".join(entries), *hints])


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
