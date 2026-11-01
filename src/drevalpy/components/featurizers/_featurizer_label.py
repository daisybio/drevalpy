"""The concat featurizer's registry name, and which featurizers must be given a view."""

from __future__ import annotations

import inspect

from drevalpy.registry.featurizer import get_featurizer
from drevalpy.types.data.feature_source import Side

CONCAT = "concatFeaturizers"
"""Registry name of the featurizer that concatenates its children; a ``+`` recipe stands for it."""


def requires_explicit_view(name: str, side: Side) -> bool:
    """Return whether a featurizer registry name requires an explicit view.

    A featurizer requires one when it declares no ``input_views`` but takes a ``view``
    argument, such as the sample ``view`` and ``pca``.

    :param name: Featurizer registry name.
    :param side: The side it is registered on.
    :returns: ``True`` for such a featurizer; ``False`` for other and unregistered names.
    """
    try:
        cls = get_featurizer(name, side)
    except (ValueError, ImportError):
        return False
    if cls.input_views:
        return False
    parameter = inspect.signature(cls.__init__).parameters.get("view")
    return parameter is not None and parameter.default is inspect.Parameter.empty
