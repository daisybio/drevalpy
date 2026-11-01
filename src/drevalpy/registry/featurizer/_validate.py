"""Registration-time checks of what a featurizer class declares."""

from __future__ import annotations

from typing import Any


def _validate_views(name: str, cls: type[Any], attr: str) -> None:
    """Raise ``TypeError`` when the view declaration *attr* is not a tuple of names.

    :param name: Registry name under which *cls* is being registered.
    :param cls: Featurizer class being registered.
    :param attr: ``input_views`` or ``source_views``.
    :raises TypeError: If the declaration is not a tuple of non-empty strings.
    """
    views = getattr(cls, attr)
    if not isinstance(views, tuple) or not all(isinstance(view, str) and view for view in views):
        msg = f"featurizer '{name}' ({cls.__name__}): {attr} must be a tuple of view names, got {views!r}"
        raise TypeError(msg)


def validate_featurizer_declarations(name: str, cls: type[Any]) -> None:
    """Check the class-body declarations of a featurizer being registered.

    A featurizer must derive from :class:`~drevalpy.components.featurizers.featurizer.Featurizer`
    and declare its views as tuples of names. Its side comes from the decorator, not the class.

    :param name: Registry name under which *cls* is being registered.
    :param cls: Featurizer class being registered.
    :raises TypeError: If *cls* is not a featurizer class or a view declaration is malformed.
    """
    from drevalpy.components.featurizers.featurizer import Featurizer

    if not (isinstance(cls, type) and issubclass(cls, Featurizer)):
        msg = f"featurizer '{name}' must be a subclass of drevalpy.components.featurizers.featurizer.Featurizer"
        raise TypeError(msg)
    _validate_views(name, cls, "input_views")
    _validate_views(name, cls, "source_views")
