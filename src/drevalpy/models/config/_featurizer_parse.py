"""Normalize featurizer mappings into canonical config fields.

Turns the mappings users write (see the YAML tab in the docs) into the plain field mappings
``FeaturizerConfig`` validates. Recipe strings are expanded by
``drevalpy.models.config._recipe`` before they reach here, so by this point a model written
as a recipe and the same model written as YAML are the same mapping and this module needs to
know nothing about recipe notation. Kept next to the config models it feeds rather than in
``drevalpy.components``, since no component consumes it.
"""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizers._featurizer_label import requires_explicit_view
from drevalpy.models.config._recipe import CONCAT_FEATURIZER_NAME, expand_featurizer_recipe
from drevalpy.models.config._space_defaults import split_space_and_options
from drevalpy.registry.cell_line_featurizer import get as get_cell_line_featurizer
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer

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


def _finalize_view(config: dict[str, Any]) -> None:
    """Settle the ``view`` of a normalized mapping, in place.

    Every notation funnels through here, so this is the single place a view is required and the
    single place an alias is resolved. Doing it here rather than while reading a recipe is what
    makes ``raw[expression]`` and a spelled-out ``view: expression`` mean the same thing.

    Only featurizers that are parametric in a view are touched.

    :param config: Normalized featurizer mapping, updated in place.
    :raises ValueError: If *config* names a view-parametric featurizer but sets no usable view.
    """
    name = str(config.get("name", ""))
    if not requires_explicit_view(name):
        return
    view = config.get("view")
    if view is None or (isinstance(view, str) and not view.strip()):
        msg = f"Featurizer {name!r} requires an explicit view, e.g. {name}[expression]"
        raise ValueError(msg)


def _featurizer_class(name: str, registry: str) -> type[Any]:
    """Look up a featurizer class in the registry the config is written against.

    :param name: Featurizer registry name.
    :param registry: ``cell_line`` or ``drug``.
    :returns: The registered featurizer class.
    """
    if registry == "cell_line":
        return get_cell_line_featurizer(name)
    return get_drug_featurizer(name)


def _assemble_featurizer_dict(
    name: str,
    *,
    default_registry: str,
    view: str | None = None,
    featurizers: list[Any] | None = None,
    hyperparameter_space: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical featurizer mapping, omitting the fields that were never set.

    :param name: Featurizer registry name.
    :param default_registry: Registry recorded on the mapping.
    :param view: View to record, when the featurizer takes one.
    :param featurizers: Normalized children, for a concat node.
    :param hyperparameter_space: Search space to record.
    :param options: Fixed constructor options to record.
    :returns: Normalized featurizer config mapping.
    """
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
    _finalize_view(payload)
    return payload


def _normalize_child(child: Any, *, default_registry: str) -> dict[str, Any]:
    """Normalize one declared child of a concat node.

    A child may be written as a recipe string, which the recipe layer expands into the mapping
    it stands for before normalization proper.

    :param child: Recipe string or mapping for one child.
    :param default_registry: Registry the child is resolved against.
    :returns: Normalized child mapping.
    """
    if isinstance(child, str):
        child = expand_featurizer_recipe(child)
    return normalize_featurizer_config(child, default_registry=default_registry)


def _normalize_child_list(children: Any, *, default_registry: str) -> list[Any]:
    """Normalize the children declared under a ``featurizers`` key.

    :param children: Value found under ``featurizers``.
    :param default_registry: Registry the children are resolved against.
    :returns: List of normalized child mappings.
    :raises ValueError: If *children* is not a list or tuple.
    """
    if isinstance(children, (str, bytes, bytearray)) or not isinstance(children, (list, tuple)):
        msg = "featurizers must be a list when set"
        raise ValueError(msg)
    return [_normalize_child(child, default_registry=default_registry) for child in children]


def _normalize_featurizer_list(data: list[Any], *, default_registry: str) -> dict[str, Any]:
    """Normalize a list of featurizers into a concat node over them.

    :param data: List of recipe strings or mappings.
    :param default_registry: Target featurizer registry name.
    :returns: Normalized concat-featurizer config mapping.
    :raises ValueError: If *data* is empty.
    """
    if not data:
        msg = "Featurizer list must be non-empty"
        raise ValueError(msg)
    return {
        "name": CONCAT_FEATURIZER_NAME,
        "featurizers": _normalize_child_list(data, default_registry=default_registry),
        "registry": default_registry,
    }


def _normalize_named_featurizer_dict(data: dict[str, Any], *, default_registry: str) -> dict[str, Any]:
    """Normalize a mapping that already names its featurizer.

    :param data: Mapping carrying at least ``name``.
    :param default_registry: Registry used when the mapping declares none.
    :returns: Normalized featurizer config mapping.
    """
    normalized = dict(data)
    normalized.setdefault("registry", default_registry)
    registry = str(normalized.get("registry", default_registry))
    if "featurizers" in normalized and normalized["featurizers"] is not None:
        normalized["featurizers"] = _normalize_child_list(
            normalized["featurizers"],
            default_registry=registry,
        )
    _finalize_view(normalized)
    return normalized


def _split_one_key_payload(
    payload: dict[str, Any],
    *,
    name: str,
    default_registry: str,
) -> tuple[list[Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Split a one-key mapping body into template fields.

    Reserved keys are taken as declared. Everything left over is a loose parameter value,
    classified against the featurizer's declared space: a tunable moves that entry's
    ``default``, anything else becomes a fixed constructor option. Explicitly declared
    entries always win over the derived ones.

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
    if body:
        derived_space, derived_options = split_space_and_options(
            _featurizer_class(name, default_registry),
            body,
        )
        if derived_space:
            hyperparameter_space = {**derived_space, **(hyperparameter_space or {})}
        if derived_options:
            options = {**derived_options, **(options or {})}
    return featurizers, hyperparameter_space, options


def _one_key_name_and_view(name_token: str) -> tuple[str, str | None]:
    """Read the key of a one-key mapping, which is written as a single recipe atom.

    A key that is not shaped like a single atom (``"a+b"``, ``"raw["``) is taken verbatim as a
    name so it fails with the registry's "unknown featurizer" error, which is what this notation
    did before.

    :param name_token: Bare featurizer name or ``name[view]`` atom.
    :returns: Featurizer name and the view written in brackets, if any.
    """
    try:
        payload = expand_featurizer_recipe(name_token)
    except ValueError:
        return name_token.strip(), None
    if payload["name"] == CONCAT_FEATURIZER_NAME:
        return name_token.strip(), None
    return str(payload["name"]), payload.get("view")


def _normalize_one_key_featurizer_dict(data: dict[str, Any], *, default_registry: str) -> dict[str, Any]:
    """Normalize the ``{"pca[methylation]": {...}}`` notation.

    :param data: Single-entry mapping of atom to arguments.
    :param default_registry: Target featurizer registry name.
    :returns: Normalized featurizer config mapping.
    :raises ValueError: If the arguments are neither ``None`` nor a mapping.
    """
    name_token, body = next(iter(data.items()))
    if body is None:
        payload: dict[str, Any] = {}
    elif isinstance(body, dict):
        payload = dict(body)
    else:
        msg = f"Featurizer {name_token!r} arguments must be a mapping when provided"
        raise ValueError(msg)
    name, view = _one_key_name_and_view(str(name_token))
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
    """Normalize a featurizer mapping into a canonical field mapping.

    Accepts a list of featurizers (equivalent to a concat node), a one-key mapping
    (``{"pca[methylation]": {...}}``), or a mapping that already has ``name``. A recipe string
    is not a mapping: callers expand one with
    ``drevalpy.models.config._recipe.expand_featurizer_recipe`` first, so that a model written
    as a recipe arrives here as the same mapping the equivalent YAML would produce.

    :param data: List of featurizers, or a field mapping.
    :param default_registry: Registry used to resolve bare names (``cell_line`` or ``drug``).
    :returns: Mapping of canonical ``FeaturizerConfig`` fields.
    :raises ValueError: If the mapping form is not a one-key shorthand and lacks ``name``.
    :raises TypeError: If *data* is not a list or mapping.
    """
    if isinstance(data, list):
        return _normalize_featurizer_list(data, default_registry=default_registry)

    if not isinstance(data, dict):
        msg = f"Featurizer config must be a list or mapping, got {type(data)!r}"
        raise TypeError(msg)

    if "name" in data:
        return _normalize_named_featurizer_dict(data, default_registry=default_registry)

    if not _RESERVED_FEATURIZER_KEYS.intersection(data.keys()) and len(data) == 1:
        return _normalize_one_key_featurizer_dict(data, default_registry=default_registry)

    msg = "Featurizer config must be a list, one-key mapping, or dict with 'name'"
    raise ValueError(msg)
