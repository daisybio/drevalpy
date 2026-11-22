"""Check a featurizer config against the constructor of its registered class.

What a config may set is read from ``__init__`` and ``input_views``, never from a list kept beside
the featurizer. The rules are written once, in :doc:`/reference/recipes`.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

#: Constructor parameters a config sets through a field of its own, not through ``options``.
_FIELDS = frozenset({"view", "featurizers"})


def requires_explicit_view(cls: type[Any]) -> bool:
    """Return whether a featurizer reads only the view its config names, so a config must name one.

    :param cls: Featurizer class.
    :returns: ``True`` when it declares no ``input_views`` and its ``view`` parameter has no default.
    """
    parameter = inspect.signature(cls).parameters.get("view")
    return not cls.input_views and parameter is not None and parameter.default is parameter.empty


def _check_view(cls: type[Any], name: str, what: str, view: str | None, *, takes_any: bool, options: str) -> None:
    """Reject a missing view the featurizer needs, or a view it cannot read.

    :param cls: Featurizer class.
    :param name: Its registry name.
    :param what: The featurizer, for the message (``"sample featurizer 'pca'"``).
    :param view: The config's ``view``, or ``None``.
    :param takes_any: Whether the constructor takes ``**kwargs``.
    :param options: The options it takes, listed for the message.
    :raises ValueError: If it needs a view and has none, takes no view, or reads only its ``input_views`` and
        *view* is none of them.
    """
    parameter = inspect.signature(cls).parameters.get("view")
    if view is None:
        if requires_explicit_view(cls):
            raise ValueError(f"Featurizer {name!r} requires an explicit view, e.g. {name}[gene_expression]")
    elif parameter is None and not takes_any:
        raise ValueError(f"{what} takes no view, but {view!r} was given; remove it. Its options: {options}")
    elif (
        cls.input_views and parameter is not None and isinstance(parameter.default, str) and view not in cls.input_views
    ):
        raise ValueError(
            f"{what} cannot read the view {view!r}; valid views: {', '.join(sorted(cls.input_views))}. "
            "To read another view unchanged, use the 'view' featurizer"
        )


def check_settings(
    cls: type[Any],
    name: str,
    side: str,
    *,
    view: str | None,
    options: Mapping[str, Any],
    space: Mapping[str, Any] | None,
) -> None:
    """Check that the view, the options and the search-space keys of a featurizer config fit its constructor.

    :param cls: The registered featurizer class.
    :param name: Its registry name.
    :param side: The side it is used on.
    :param view: The config's ``view``, or ``None``.
    :param options: The config's ``options``.
    :param space: The config's own ``hyperparameter_space``, or ``None`` when it declares none.
    :raises ValueError: If ``view`` is written under ``options``, an option or search-space key is not a
        constructor parameter, the featurizer takes no view or not this one, or it needs a view and has none.
        The message lists the valid choices.
    """
    what = f"{side} featurizer {name!r}"
    if "view" in options:
        raise ValueError(
            f"{what} takes 'view' as a key next to 'name', not under 'options'; "
            f"write {{name: {name}, view: {options['view']}}}"
        )
    parameters = inspect.signature(cls).parameters
    takes_any = any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values())
    keys = sorted(
        key
        for key, parameter in parameters.items()
        if key not in _FIELDS and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    )
    valid = ", ".join(keys) or "none"
    for field, given in (("options", options), ("hyperparameter_space", space or {})):
        unknown = sorted(set(given) - set(keys))
        if unknown and not takes_any:
            raise ValueError(f"{field} of {what} name unknown keys ({', '.join(unknown)}); valid keys: {valid}")
    _check_view(cls, name, what, view, takes_any=takes_any, options=valid)
