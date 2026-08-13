"""Public register / get / list / table / metadata helpers for the predictor registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.registry.predictor._registry import predictor_registry
from drevalpy.types.enums.literature_reference import LiteratureReference


def register(
    name: str,
    *,
    description: str,
    cell_line_contract: FeatureContract | FeatureFormat | None = None,
    drug_contract: FeatureContract | FeatureFormat | None = None,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
) -> Callable[[type[Any]], type[Any]]:
    """Decorator: register a predictor.

    :param name: Registry name used in ``ModelConfig`` and recipes.
    :param description: Short human-readable summary for catalog listings.
    :param cell_line_contract: Expected cell-line feature format. Falls back to the
        ``cell_line_contract`` declared on the class body when omitted.
    :param drug_contract: Expected drug feature format. Falls back to the
        ``drug_contract`` declared on the class body when omitted.
    :param tags: Optional discovery tags.
    :param reference: Optional literature citation metadata.

    :returns: Class decorator that registers the decorated predictor under *name*.
    """
    return predictor_registry.register(
        name,
        description=description,
        tags=tags,
        reference=reference,
        cell_line_contract=cell_line_contract,
        drug_contract=drug_contract,
    )


def get(name: str) -> type[Any]:
    """Return the predictor class registered under *name*.

    :param name: Registry name of the predictor.
    :returns: Predictor class registered under *name*.
    :raises ValueError: If *name* is not registered.
    """
    return predictor_registry.get(name)


def list() -> list[str]:  # noqa: A001
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
