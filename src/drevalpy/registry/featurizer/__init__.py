"""Shared featurizer registry infrastructure used by sample and drug sub-packages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from drevalpy.errors import UnknownNameError

from ._base import FeaturizerRegistry
from ._validate import validate_featurizer_input_views

if TYPE_CHECKING:
    from drevalpy.components.featurizers.base import Featurizer

__all__ = ["FeaturizerRegistry", "Side", "get_featurizer", "validate_featurizer_input_views"]

#: The entity side a featurizer is registered for.
Side = Literal["sample", "drug"]


def get_featurizer(side: Side, name: str) -> type[Featurizer]:
    """Return the featurizer class registered under *name* for one entity side.

    The side packages are imported inside the function because they import the component
    contracts, which in turn import this package.

    :param side: ``"sample"`` or ``"drug"``.
    :param name: Registry name of the featurizer.
    :returns: Featurizer class registered under *name* on that side.
    :raises UnknownNameError: If *side* is not ``"sample"`` or ``"drug"``, or *name* is not registered on it.
    """
    if side == "sample":
        from drevalpy.registry.sample_featurizer import get as get_sample

        return get_sample(name)
    if side == "drug":
        from drevalpy.registry.drug_featurizer import get as get_drug

        return get_drug(name)
    raise UnknownNameError(f"Unknown featurizer side {side!r}. Available: ['sample', 'drug']")
