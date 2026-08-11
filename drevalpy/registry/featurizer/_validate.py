"""Registration-time invariants for featurizer input-view declarations."""

from __future__ import annotations

from typing import Any


def _declares_input_views(cls: type[Any]) -> bool:
    """Return whether *cls* states which raw views it reads.

    :param cls: Featurizer class being registered.
    :returns: ``True`` when the class declares, parameterizes, or computes its input views.
    """
    from drevalpy.components.featurizers.base import Featurizer

    if cls.input_views is not None:
        return True
    if cls.requires_view or cls.entity_id_only:
        return True
    return getattr(cls.resolve_input_views, "__func__", None) is not getattr(
        Featurizer.resolve_input_views, "__func__", None
    )


def validate_featurizer_input_views(registry_id: str, name: str, cls: type[Any]) -> None:
    """Raise ``ValueError`` when a featurizer does not declare its input views.

    Every featurizer must state which raw feature views it consumes so that the
    data-loading layer never needs a name-to-view lookup table. Classes that do not
    derive from ``Featurizer`` are skipped; they do not participate in view resolution.

    :param registry_id: Registry identifier used in the error message.
    :param name: Registry name under which *cls* is being registered.
    :param cls: Featurizer class being registered.
    :raises ValueError: If the class declares no input views.
    """
    from drevalpy.components.featurizers.base import Featurizer

    if not (isinstance(cls, type) and issubclass(cls, Featurizer)):
        return
    if _declares_input_views(cls):
        return
    msg = (
        f"{registry_id} '{name}' ({cls.__name__}) does not declare its input views. "
        "Set input_views on the class body (use () when only entity ids are needed), "
        "set requires_view/entity_id_only, or override resolve_input_views."
    )
    raise ValueError(msg)
