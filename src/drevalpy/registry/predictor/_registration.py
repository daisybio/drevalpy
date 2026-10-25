"""Public register / get / list / table / metadata helpers for the predictor registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.registry.components._metadata import LiteratureReference
from drevalpy.registry.predictor._registry import predictor_registry

if TYPE_CHECKING:
    import pandas as pd


def register(
    name: str,
    *,
    description: str,
    sample_input_format: FeatureFormat | None = None,
    drug_input_format: FeatureFormat | None = None,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
) -> Callable[[type[Any]], type[Any]]:
    """Return a decorator that registers a predictor.

    :param name: Registry name used in ``ModelConfig`` and recipes.
    :param description: Short human-readable summary for catalog listings.
    :param sample_input_format: Expected sample feature format. Falls back to the
        ``sample_input_format`` declared on the class body when omitted.
    :param drug_input_format: Expected drug feature format. Falls back to the
        ``drug_input_format`` declared on the class body when omitted.
    :param tags: Optional discovery tags.
    :param reference: Optional literature citation metadata.

    :returns: Class decorator that registers the decorated predictor under *name*.
    """
    return predictor_registry.register(
        name,
        description=description,
        tags=tags,
        reference=reference,
        sample_input_format=sample_input_format,
        drug_input_format=drug_input_format,
    )


def get(name: str) -> type[Any]:
    """Return the predictor class registered under *name*.

    Raises ``UnknownNameError`` (a ``ValueError``) if *name* is not registered.

    :param name: Registry name of the predictor.
    :returns: Predictor class registered under *name*.
    """
    return predictor_registry.get(name)


def list() -> list[str]:
    """Return sorted list of registered predictor names.

    :returns: All currently registered predictor names.
    """
    return predictor_registry.list_names()


def table() -> pd.DataFrame:
    """Return registry contents as a DataFrame.

    :returns: DataFrame with Name, Description, and Tags columns.
    """
    return predictor_registry.to_dataframe()


def metadata(name: str) -> dict[str, Any]:
    """Return metadata for a registered predictor.

    :param name: Registry name of the predictor.
    :returns: Metadata dict including input interface, tags, and literature fields.
    """
    return predictor_registry.get_metadata(name)
