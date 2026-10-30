"""Normalize every accepted spelling of a featurizer into the canonical ``FeaturizerConfig`` fields.

A featurizer slot accepts six spellings; :func:`normalize_featurizer` is the one place that reads
them::

    "scaledGeneExpression"                     name=scaledGeneExpression
    "raw[gene_expression]"                     name=raw, view=gene_expression
    "raw[gene_expression]+raw[mutations]"      name=concatFeaturizers, two children
    ["raw[gene_expression]", "raw[mutations]"] name=concatFeaturizers, two children
    {"pca[methylation]": {"n_components": 8}}  name=pca, view=methylation, + space
    {"name": "raw", "view": "gene_expression"} already canonical, passed through

Recipe strings are read by ``_recipe.expand_featurizer_recipe``, so a model written as a recipe
and the same model written as YAML give the same mapping. A loose value in the one-key mapping is
classified against the component's declared space by :func:`split_loose_values`: a key the space
declares moves that entry's ``default``, anything else is a fixed constructor option.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from drevalpy.models.config._recipe import CONCAT_FEATURIZER_NAME, expand_featurizer_recipe
from drevalpy.registry.featurizer import get_featurizer

_RESERVED_KEYS = frozenset({"name", "hyperparameters", "featurizers", "view", "hyperparameter_space", "options"})


def split_loose_values(cls: type[Any], values: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split loose parameter values into space-default overrides and fixed options.

    The returned space is the component's full declared space with the matched defaults
    replaced, not only the touched entries, so the config records a complete space.

    :param cls: Component class exposing ``get_hyperparameter_space``.
    :param values: Mapping of local parameter name to concrete value.
    :returns: ``(hyperparameter_space, options)``.
    """
    space = {
        key: dict(spec) if isinstance(spec, dict) else spec for key, spec in cls.get_hyperparameter_space().items()
    }
    options: dict[str, Any] = {}
    for key, value in values.items():
        if key in space and isinstance(space[key], dict):
            space[key] = {**space[key], "default": value}
        else:
            options[key] = value
    return space, options


def _atom(token: str) -> tuple[str, str | None]:
    """Read the key of a one-key mapping as a single recipe atom, ``name`` or ``name[view]``.

    A key that is not one atom (``"a+b"``, ``"raw["``) is taken verbatim as a name, so it fails
    with the registry's "unknown featurizer" error.
    """
    try:
        payload = expand_featurizer_recipe(token)
    except ValueError:
        return token.strip(), None
    if payload["name"] == CONCAT_FEATURIZER_NAME:
        return token.strip(), None
    return str(payload["name"]), payload.get("view")


def _one_key_fields(token: str, body: Any) -> dict[str, Any]:
    """Read the ``{"pca[methylation]": {...}}`` spelling into canonical fields."""
    if body is not None and not isinstance(body, dict):
        raise ValueError(f"Featurizer {token!r} arguments must be a mapping when provided")
    loose = dict(body or {})
    name, view = _atom(token)
    fields: dict[str, Any] = {"name": name}
    if view is not None:
        fields["view"] = view
    for reserved in ("featurizers", "hyperparameter_space", "options"):
        if reserved in loose:
            fields[reserved] = loose.pop(reserved)
    loose.pop("view", None)
    if loose:
        space, options = split_loose_values(get_featurizer(name), loose)
        if space:
            fields["hyperparameter_space"] = {**space, **(fields.get("hyperparameter_space") or {})}
        if options:
            fields["options"] = {**options, **(fields.get("options") or {})}
    return fields


def normalize_featurizer(raw: Any) -> dict[str, Any]:
    """Return the canonical ``FeaturizerConfig`` fields of any accepted featurizer spelling.

    Children of a concat node are normalized the same way.

    :param raw: Recipe string, list of featurizers, one-key mapping, or canonical mapping.
    :returns: Canonical field mapping.
    :raises ValueError: If *raw* is an empty list or a mapping in none of the accepted shapes.
    :raises TypeError: If *raw* is not a string, list or mapping.
    """
    if isinstance(raw, str):
        fields = expand_featurizer_recipe(raw)
    elif isinstance(raw, list):
        if not raw:
            raise ValueError("Featurizer list must be non-empty")
        fields = {"name": CONCAT_FEATURIZER_NAME, "featurizers": raw}
    elif not isinstance(raw, dict):
        raise TypeError(f"Featurizer config must be a string, list or mapping, got {type(raw)!r}")
    elif "name" in raw:
        fields = dict(raw)
    elif len(raw) == 1 and not _RESERVED_KEYS.intersection(raw):
        token, body = next(iter(raw.items()))
        fields = _one_key_fields(str(token), body)
    else:
        raise ValueError("Featurizer config must be a string, list, one-key mapping, or dict with 'name'")
    children = fields.get("featurizers")
    if children is not None:
        if not isinstance(children, (list, tuple)):
            raise ValueError("featurizers must be a list when set")
        fields["featurizers"] = [normalize_featurizer(child) for child in children]
    return fields
