"""Classify loose parameter values against a component's declared hyperparameter space.

Both compact config notations let a value be written next to the component name rather than
inside a full search-space spec (``{"pca[methylation]": {"n_components": 8}}``). Deciding
what such a value means needs the component's own declared space: a key it declares is a
tunable whose ``default`` moves, anything else is a fixed constructor option.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def split_space_and_options(
    cls: type[Any],
    values: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split loose parameter values into space-default overrides and fixed options.

    The returned space is the component's full declared space with the matched defaults
    replaced, not only the touched entries, so the config records a complete space. Callers
    apply their own empty-versus-``None`` convention to both halves.

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
