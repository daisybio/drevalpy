"""The concat featurizer's registry name, and which featurizers must be given a view."""

from __future__ import annotations

import inspect

from drevalpy.registry.featurizer import get_featurizer

CONCAT = "concatFeaturizers"
"""Registry name of the featurizer that concatenates its children; a ``+`` recipe stands for it."""


def requires_explicit_view(name: str) -> bool:
    """Return whether a featurizer registry name requires an explicit view.

    A featurizer requires one when it declares no ``input_views`` but takes a ``view``
    argument, such as ``raw`` and ``pca``.

    :param name: Featurizer registry name.
    :returns: ``True`` for such a featurizer; ``False`` for other and unregistered names.
    """
    try:
        cls = get_featurizer(name)
    except (ValueError, ImportError):
        return False
    if cls.input_views:
        return False
    parameter = inspect.signature(cls.__init__).parameters.get("view")
    return parameter is not None and parameter.default is inspect.Parameter.empty
