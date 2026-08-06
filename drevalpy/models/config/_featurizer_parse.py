"""Normalize featurizer recipe strings and mappings into canonical config fields.

Turns the notations users write (see the recipe-string and YAML tabs in the docs) into
the plain field mappings ``FeaturizerConfig`` validates. Kept next to the config models
it feeds rather than in ``drevalpy.components``, since no component consumes it.
"""

from __future__ import annotations

import re
from typing import Any

from drevalpy.components.featurizer_label import requires_explicit_view
from drevalpy.components.view_aliases import resolve_omics_view

_RESERVED_FEATURIZER_KEYS = frozenset(
    {
        "name",
        "hyperparameters",
        "featurizers",
        "registry",
        "view",
        "hyperparameter_space",
        "options",
    }
)
_CONCAT_FEATURIZER_NAME = "concatFeaturizers"
_BRACKET_ATOM_RE = re.compile(r"^([^[\]]+)\[([^\]]+)\]$")


def _split_concat_recipe(token: str) -> list[str]:
    """Split a concat recipe on ``+`` outside square brackets.

    Bracket depth is tracked so a ``+`` inside a view (``raw[a+b]``) does not split the
    recipe, which keeps the error pointing at the bad view rather than at a truncated name.

    :param token: Featurizer recipe string that may join atoms with ``+``.
    :returns: Non-empty recipe segments outside bracket nesting.
    :raises ValueError: If ``+`` appears at boundaries or consecutively.
    """
    if token.startswith("+") or token.endswith("+") or "++" in token:
        msg = "Featurizer recipe segments joined by '+' must be non-empty"
        raise ValueError(msg)
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(token):
        depth += (char == "[") - (char == "]")
        if char == "+" and depth == 0:
            parts.append(token[start:index])
            start = index + 1
    parts.append(token[start:])
    return [part.strip() for part in parts if part.strip()]


def _parse_bracket_atom_name(name_token: str, *, default_registry: str) -> tuple[str, str | None]:
    """Parse ``name[view]`` into registry name and resolved view.

    :param name_token: Bare featurizer name or ``name[view]`` atom.
    :param default_registry: Registry context (``cell_line`` or ``drug``).
    :returns: Registry name and resolved view, or ``(name, None)`` when unbracketed.
    :raises ValueError: If bracket syntax is used for unsupported featurizers or registries.
    """
    match = _BRACKET_ATOM_RE.match(name_token.strip())
    if not match:
        return name_token.strip(), None
    name, view_token = match.groups()
    name = name.strip()
    if not requires_explicit_view(name):
        msg = f"Bracket syntax is only supported for raw and pca, got {name!r}"
        raise ValueError(msg)
    if default_registry != "cell_line":
        msg = f"Bracket view syntax is only supported for cell-line featurizers, got registry {default_registry!r}"
        raise ValueError(msg)
    return name, resolve_omics_view(view_token)


def _validate_view_required(config: dict[str, Any]) -> None:
    name = str(config.get("name", ""))
    if not requires_explicit_view(name):
        return
    view = config.get("view")
    if view is None or (isinstance(view, str) and not view.strip()):
        msg = f"Featurizer {name!r} requires an explicit view, e.g. {name}[expression]"
        raise ValueError(msg)


def _lift_featurizers_from_hyperparameters(
    normalized: dict[str, Any],
    leftover: dict[str, Any],
) -> None:
    if "featurizers" not in leftover:
        return
    if "featurizers" in normalized and normalized["featurizers"] is not None:
        msg = "Featurizer config cannot set both 'featurizers' and hyperparameters['featurizers']"
        raise ValueError(msg)
    normalized["featurizers"] = leftover.pop("featurizers")


def _fold_simple_values(
    name: str,
    simple_values: dict[str, Any],
    *,
    registry: str,
    hyperparameter_space: dict[str, Any] | None,
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Fold loose parameter values into a tuning space and fixed constructor options.

    Shared by both mapping notations: a one-key body writes its parameters alongside the
    reserved keys, while the legacy form nests them under ``hyperparameters``. Once the
    caller has separated the two, the merge rule is the same, so it lives here only.
    Explicitly declared entries always win over values derived from *simple_values*.

    :param name: Featurizer registry name, used to look up the declared space.
    :param simple_values: Loose ``parameter: value`` pairs to classify.
    :param registry: ``cell_line`` or ``drug``.
    :param hyperparameter_space: Space the config declared explicitly, if any.
    :param options: Options the config declared explicitly, if any.
    :returns: ``(hyperparameter_space, options)`` with derived values merged underneath.
    """
    if not simple_values:
        return hyperparameter_space, options
    derived_space, derived_options = _space_defaults_from_simple_values(
        name,
        simple_values,
        default_registry=registry,
    )
    if derived_space:
        hyperparameter_space = {**derived_space, **(hyperparameter_space or {})}
    if derived_options:
        options = {**derived_options, **(options or {})}
    return hyperparameter_space, options


def _lift_legacy_hyperparameters(config: dict[str, Any], *, default_registry: str) -> dict[str, Any]:
    """Move legacy ``hyperparameters`` into ``featurizers`` / ``options`` / space defaults.

    :param config: Partially normalized featurizer mapping.
    :param default_registry: Registry used when converting tunable shorthand values.
    :returns: Mapping without a ``hyperparameters`` key.
    :raises ValueError: If ``hyperparameters`` is malformed.
    """
    normalized = dict(config)
    hyperparameters = normalized.pop("hyperparameters", None)
    if hyperparameters is None:
        return normalized
    if not isinstance(hyperparameters, dict):
        msg = "Featurizer hyperparameters must be a mapping when provided"
        raise ValueError(msg)
    if not hyperparameters:
        return normalized
    leftover = dict(hyperparameters)
    _lift_featurizers_from_hyperparameters(normalized, leftover)
    space, options = _fold_simple_values(
        str(normalized.get("name", "")),
        leftover,
        registry=str(normalized.get("registry", default_registry)),
        hyperparameter_space=normalized.get("hyperparameter_space"),
        options=normalized.get("options"),
    )
    if space is not None:
        normalized["hyperparameter_space"] = space
    if options is not None:
        normalized["options"] = options
    return normalized


def _finalize_featurizer_config(config: dict[str, Any], *, default_registry: str = "cell_line") -> dict[str, Any]:
    normalized = _lift_legacy_hyperparameters(config, default_registry=default_registry)
    _validate_view_required(normalized)
    return normalized


def _assemble_featurizer_dict(
    name: str,
    *,
    default_registry: str,
    view: str | None = None,
    featurizers: list[Any] | None = None,
    hyperparameter_space: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "registry": default_registry,
    }
    if view is not None:
        payload["view"] = view
    if featurizers is not None:
        payload["featurizers"] = featurizers
    if hyperparameter_space is not None:
        payload["hyperparameter_space"] = hyperparameter_space
    if options is not None:
        payload["options"] = options
    return _finalize_featurizer_config(payload, default_registry=default_registry)


def _require_view_for_parametric(name: str, view: str | None) -> None:
    if view is None and requires_explicit_view(name):
        msg = f"Featurizer {name!r} requires an explicit view, e.g. {name}[expression]"
        raise ValueError(msg)


def _parse_featurizer_atom(token: str, *, default_registry: str) -> dict[str, Any]:
    """Normalize one featurizer atom, including optional ``name[view]`` syntax.

    :param token: Single featurizer atom from a concat recipe.
    :param default_registry: Target featurizer registry name.
    :returns: Normalized featurizer config mapping.
    :raises ValueError: If the atom is empty or requires a missing view.
    """
    trimmed = token.strip()
    if not trimmed:
        msg = "Featurizer token must be a non-empty string"
        raise ValueError(msg)
    name, view = _parse_bracket_atom_name(trimmed, default_registry=default_registry)
    _require_view_for_parametric(name, view)
    return _assemble_featurizer_dict(
        name,
        default_registry=default_registry,
        view=view,
    )


def _parse_featurizer_token(token: str, *, default_registry: str) -> dict[str, Any]:
    """Normalize a bare featurizer token, including ``+`` concat recipes.

    :param token: String featurizer recipe from a model config.
    :param default_registry: Target featurizer registry name.
    :returns: Normalized featurizer or concat-featurizer config mapping.
    :raises ValueError: If the token is empty or contains invalid concat syntax.
    """
    trimmed = token.strip()
    if not trimmed:
        msg = "Featurizer token must be a non-empty string"
        raise ValueError(msg)
    parts = _split_concat_recipe(trimmed)
    if len(parts) == 1:
        return _parse_featurizer_atom(parts[0], default_registry=default_registry)

    if any(not part for part in parts):
        msg = "Featurizer recipe segments joined by '+' must be non-empty"
        raise ValueError(msg)

    return {
        "name": _CONCAT_FEATURIZER_NAME,
        "featurizers": [_parse_featurizer_atom(part, default_registry=default_registry) for part in parts],
        "registry": default_registry,
    }


def _normalize_featurizer_list(data: list[Any], *, default_registry: str) -> dict[str, Any]:
    if not data:
        msg = "Featurizer list must be non-empty"
        raise ValueError(msg)
    return {
        "name": _CONCAT_FEATURIZER_NAME,
        "featurizers": [normalize_featurizer_config(item, default_registry=default_registry) for item in data],
        "registry": default_registry,
    }


def _normalize_child_list(children: Any, *, default_registry: str) -> list[Any]:
    if isinstance(children, (str, bytes, bytearray)) or not isinstance(children, (list, tuple)):
        msg = "featurizers must be a list when set"
        raise ValueError(msg)
    return [normalize_featurizer_config(item, default_registry=default_registry) for item in children]


def _normalize_named_featurizer_dict(data: dict[str, Any], *, default_registry: str) -> dict[str, Any]:
    normalized = dict(data)
    normalized.setdefault("registry", default_registry)
    registry = str(normalized.get("registry", default_registry))
    if "featurizers" in normalized and normalized["featurizers"] is not None:
        normalized["featurizers"] = _normalize_child_list(
            normalized["featurizers"],
            default_registry=registry,
        )
    return _finalize_featurizer_config(normalized, default_registry=registry)


def _space_defaults_from_simple_values(
    name: str,
    simple_values: dict[str, Any],
    *,
    default_registry: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Split shorthand values into space-default overrides and fixed options.

    Tunable keys update ``hyperparameter_space`` defaults. Non-tunable keys become
    template ``options`` passed through to the featurizer constructor.

    :param name: Featurizer registry name.
    :param simple_values: Mapping of local parameter name to concrete value.
    :param default_registry: ``cell_line`` or ``drug``.
    :returns: ``(hyperparameter_space, options)``.
    """
    from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer

    cls = get_cell_line_featurizer(name) if default_registry == "cell_line" else get_drug_featurizer(name)
    space = {
        key: dict(spec) if isinstance(spec, dict) else spec for key, spec in cls.get_hyperparameter_space().items()
    }
    options: dict[str, Any] = {}
    for key, value in simple_values.items():
        if key in space and isinstance(space[key], dict):
            space[key] = {**space[key], "default": value}
        else:
            options[key] = value
    return (space if space else None), (options or None)


def _split_one_key_payload(
    payload: dict[str, Any],
    *,
    name: str,
    default_registry: str,
) -> tuple[list[Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Split a one-key mapping body into template fields.

    Reserved keys are taken as declared; everything left over is a loose parameter value
    folded in by ``_fold_simple_values``.

    :param payload: Body of a one-key featurizer mapping.
    :param name: Featurizer registry name.
    :param default_registry: ``cell_line`` or ``drug``.
    :returns: ``(featurizers, hyperparameter_space, options)``.
    """
    body = dict(payload)
    featurizers = body.pop("featurizers", None)
    hyperparameter_space = body.pop("hyperparameter_space", None)
    options = body.pop("options", None)
    body.pop("view", None)
    hyperparameter_space, options = _fold_simple_values(
        name,
        body,
        registry=default_registry,
        hyperparameter_space=hyperparameter_space,
        options=options,
    )
    return featurizers, hyperparameter_space, options


def _normalize_one_key_featurizer_dict(data: dict[str, Any], *, default_registry: str) -> dict[str, Any]:
    name_token, body = next(iter(data.items()))
    if body is None:
        payload: dict[str, Any] = {}
    elif isinstance(body, dict):
        payload = dict(body)
    else:
        msg = f"Featurizer {name_token!r} arguments must be a mapping when provided"
        raise ValueError(msg)
    name, view = _parse_bracket_atom_name(str(name_token), default_registry=default_registry)
    _require_view_for_parametric(name, view)
    featurizers, hyperparameter_space, options = _split_one_key_payload(
        payload,
        name=name,
        default_registry=default_registry,
    )
    if featurizers is not None:
        featurizers = _normalize_child_list(featurizers, default_registry=default_registry)
    return _assemble_featurizer_dict(
        name,
        default_registry=default_registry,
        view=view,
        featurizers=featurizers,
        hyperparameter_space=hyperparameter_space,
        options=options,
    )


def normalize_featurizer_config(data: Any, *, default_registry: str = "cell_line") -> dict[str, Any]:
    """Normalize any accepted featurizer notation into a canonical field mapping.

    Accepts a recipe string (``"raw[gene_expression]"``, or ``+``-joined atoms), a list of
    those (equivalent to a concat node), a one-key mapping (``{"pca[methylation]": {...}}``),
    or a mapping that already has ``name``.

    :param data: Recipe string, list of recipes, or field mapping.
    :param default_registry: Registry used to resolve bare names (``cell_line`` or ``drug``).
    :returns: Mapping of canonical ``FeaturizerConfig`` fields.
    :raises ValueError: If the mapping form is not a one-key shorthand and lacks ``name``.
    :raises TypeError: If *data* is not a string, list, or mapping.
    """
    if isinstance(data, str):
        return _parse_featurizer_token(data, default_registry=default_registry)

    if isinstance(data, list):
        return _normalize_featurizer_list(data, default_registry=default_registry)

    if not isinstance(data, dict):
        msg = f"Featurizer config must be a string, list, or mapping, got {type(data)!r}"
        raise TypeError(msg)

    if "name" in data:
        return _normalize_named_featurizer_dict(data, default_registry=default_registry)

    if not _RESERVED_FEATURIZER_KEYS.intersection(data.keys()) and len(data) == 1:
        return _normalize_one_key_featurizer_dict(data, default_registry=default_registry)

    msg = "Featurizer config must be a string, one-key mapping, or dict with 'name'"
    raise ValueError(msg)
