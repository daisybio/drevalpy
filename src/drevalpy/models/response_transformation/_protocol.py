"""The structural type every response transformation satisfies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ResponseTransformer(Protocol):
    """An unfitted or fitted response transformation, such as a sklearn scaler or ``GroupMeanCenterer``.

    The members are loosely typed because sklearn's scalers and ``GroupMeanCenterer`` take different
    arguments. ``get_params`` and ``set_params`` are what ``sklearn.base.clone`` needs. A transformer
    that needs group labels also sets ``requires_groups = True`` and names its group ``levels``.
    """

    fit: Callable[..., Any]
    transform: Callable[..., Any]
    inverse_transform: Callable[..., Any]
    get_params: Callable[..., Any]
    set_params: Callable[..., Any]
