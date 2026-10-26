"""Response transformations: selection by name, ``GroupMeanCenterer`` and the pipeline helpers.

``GroupMeanCenterer`` subclasses sklearn's ``BaseEstimator``, so it is imported on first access;
``import drevalpy`` must not load sklearn (see ``tests/test_import_cost_policy.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._names import get_response_transformation
from ._protocol import ResponseTransformer

if TYPE_CHECKING:
    from ._group_mean import GroupMeanCenterer

__all__ = ["GroupMeanCenterer", "ResponseTransformer", "get_response_transformation"]


def __getattr__(name: str) -> Any:
    """Import ``GroupMeanCenterer`` on first access.

    :param name: Attribute name.
    :returns: The ``GroupMeanCenterer`` class.
    :raises AttributeError: For any other name.
    """
    if name == "GroupMeanCenterer":
        from ._group_mean import GroupMeanCenterer

        return GroupMeanCenterer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
