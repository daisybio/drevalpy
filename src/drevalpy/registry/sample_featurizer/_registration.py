"""Public register / get / list / table / metadata helpers for the sample featurizer registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.types.enums.literature_reference import LiteratureReference

from ._registry import sample_featurizer_registry

if TYPE_CHECKING:
    import pandas as pd


def register(
    name: str,
    *,
    description: str,
    contract: FeatureContract | FeatureFormat | None = None,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
) -> Callable[[type[Any]], type[Any]]:
    """Decorator: register a sample featurizer.

    :param name: Registry name used in ``ModelConfig`` and recipes.
    :param description: Short human-readable summary for catalog listings.
    :param contract: Feature format contract for predictor compatibility checks.
        Falls back to the ``contract`` declared on the class body when omitted.
    :param tags: Optional discovery tags (for example ``"omics"``).
    :param reference: Optional literature citation metadata.
    :returns: Class decorator that registers the decorated featurizer under *name*.
    """
    return sample_featurizer_registry.register(
        name,
        description=description,
        contract=contract,
        tags=tags,
        reference=reference,
    )


def get(name: str) -> type[Any]:
    """Return the sample featurizer class registered under *name*.

    :param name: Registry name of the featurizer.
    :returns: Featurizer class registered under *name*.
    :raises ValueError: If *name* is not registered.
    """
    return sample_featurizer_registry.get(name)


def list() -> list[str]:
    """Return sorted list of registered sample featurizer names."""
    return sample_featurizer_registry.list_names()


def table() -> pd.DataFrame:
    """Return registry contents as a DataFrame."""
    return sample_featurizer_registry.to_dataframe()


def metadata(name: str) -> dict[str, Any]:
    """Return metadata for a registered sample featurizer.

    :param name: Registry name of the featurizer.
    :returns: Metadata dict including output format and tags.
    """
    return sample_featurizer_registry.get_metadata(name)
