"""What a featurizer config may set, read from the constructor of the registered class.

A featurizer takes its settings as keyword arguments of ``__init__``: the ``view`` field of a
config, its ``options`` and the keys of its search space all end up there. So what a config
may name is a fact of the class, and :func:`check_settings` reads it from the signature and
from ``input_views``, never from a list kept beside the featurizer. A config is checked when it
is built, so a typo fails with the valid choices instead of as a ``TypeError`` when the model
trains.

* ``view`` needs a ``view`` parameter, and goes next to ``name`` rather than under ``options``;
  ``oneHot[x]`` and ``molirOmics[x]`` name a view their featurizer cannot take.
* A ``view`` parameter with a named default, such as ``scaledGeneExpression``'s
  ``"gene_expression"``, reads one of the featurizer's ``input_views``; the config may name any
  of them, nothing else. A ``view`` parameter that is required (``pca``, sample ``view``) or
  defaults to ``None`` (drug ``view``) reads whatever view the config names.
* An option or search-space key must be a parameter. A constructor with ``**kwargs`` accepts
  any key, so it is not checked.

Whether the dataset holds the view is not known here, and is checked when the model trains.
"""

from __future__ import annotations

import inspect
from collections.abc import Collection, Mapping
from typing import Any

from drevalpy.components.hyperparameter_space import effective_space

#: Constructor parameters a config sets through a field of its own, not through ``options``.
_FIELDS = frozenset({"view", "featurizers"})


def _parameters(cls: type[Any]) -> tuple[Mapping[str, inspect.Parameter], bool]:
    """Return the keyword parameters of a featurizer's constructor and whether it takes ``**kwargs``.

    :param cls: Featurizer class.
    :returns: The named parameters except ``self``, and whether any keyword is accepted.
    """
    if cls.__init__ is object.__init__:
        return {}, False
    parameters = inspect.signature(cls.__init__).parameters
    named = {
        name: parameter
        for name, parameter in parameters.items()
        if name != "self" and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    }
    return named, any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values())


def _choices(names: Collection[str]) -> str:
    """List *names* sorted, or ``none`` when there are none.

    :param names: Valid names.
    :returns: The names joined by commas.
    """
    return ", ".join(sorted(names)) or "none"


def _check_keys(cls: type[Any], what: str, field: str, keys: Collection[str], valid: Collection[str]) -> None:
    """Reject the keys of ``options`` or of a search space that the constructor has no parameter for.

    :param cls: Featurizer class.
    :param what: The featurizer, for the message (``"sample featurizer 'pca'"``).
    :param field: Where the keys are written (``"options"`` or ``"hyperparameter_space"``).
    :param keys: The keys.
    :param valid: The keys that may be written there.
    :raises ValueError: If a key is not in *valid* and the constructor does not take ``**kwargs``.
    """
    unknown = sorted(set(keys) - set(valid))
    if unknown and not _parameters(cls)[1]:
        raise ValueError(f"{field} of {what} name unknown keys ({', '.join(unknown)}); valid keys: {_choices(valid)}")


def _check_view(cls: type[Any], what: str, view: str) -> None:
    """Reject a view the featurizer cannot read.

    :param cls: Featurizer class.
    :param what: The featurizer, for the message.
    :param view: The configured view.
    :raises ValueError: If the featurizer has no ``view`` parameter, or reads only its input views and *view* is
        none of them.
    """
    parameters, takes_any = _parameters(cls)
    parameter = parameters.get("view")
    if parameter is None and not takes_any:
        raise ValueError(
            f"{what} takes no view, but {view!r} was given; remove it. "
            f"Its options: {_choices(parameters.keys() - _FIELDS)}"
        )
    if parameter is not None and cls.input_views and isinstance(parameter.default, str) and view not in cls.input_views:
        raise ValueError(
            f"{what} cannot read the view {view!r}; valid views: {_choices(cls.input_views)}. "
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
    :raises ValueError: If the featurizer takes no view or not this one, ``view`` is written under ``options``, or
        an option or search-space key is not a constructor parameter. The message lists the valid choices.
    """
    what = f"{side} featurizer {name!r}"
    if "view" in options:
        raise ValueError(
            f"{what} takes 'view' as a key next to 'name', not under 'options'; "
            f"write {{name: {name}, view: {options['view']}}}"
        )
    parameters = _parameters(cls)[0].keys() - _FIELDS
    tunable = {key for key, spec in effective_space(space, cls).items() if isinstance(spec, Mapping)}
    _check_keys(cls, what, "options", options, parameters - tunable)
    _check_keys(cls, what, "hyperparameter_space", space or {}, parameters)
    if view is not None:
        _check_view(cls, what, view)
