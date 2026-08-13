"""Registration-time rejection of classes with unimplemented abstract methods.

A component that forgets ``_fit`` registers happily and only fails when the
experiment instantiates it, far from the cause. The registries therefore reject
such a class up front, naming the members it still has to implement.
"""

from __future__ import annotations

from typing import Any


def abstract_members(cls: type[Any]) -> tuple[str, ...]:
    """Return the abstract members *cls* has not implemented.

    :param cls: Component class being registered.
    :returns: Sorted member names still declared abstract, empty when the class
        is concrete (or is not an ABC at all).
    """
    return tuple(sorted(getattr(cls, "__abstractmethods__", frozenset())))


def validate_no_abstract_methods(registry_id: str, name: str, cls: type[Any]) -> None:
    """Raise ``ValueError`` when *cls* still has unimplemented abstract members.

    :param registry_id: Registry identifier used in the error message.
    :param name: Registry name under which *cls* is being registered.
    :param cls: Component class being registered.
    :raises ValueError: If the class cannot be instantiated because abstract
        members remain unimplemented.
    """
    missing = abstract_members(cls)
    if not missing:
        return
    msg = (
        f"{registry_id} '{name}' ({cls.__name__}) does not implement "
        f"{', '.join(missing)}. Implement the missing member(s), or register a "
        "concrete subclass instead of an abstract base."
    )
    raise ValueError(msg)
