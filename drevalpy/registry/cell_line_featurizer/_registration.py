"""Public register / get / list / table / metadata helpers for the cell-line featurizer registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.types.enums.literature_reference import LiteratureReference

from ._registry import cell_line_featurizer_registry


def register(
    name: str,
    *,
    description: str,
    contract: FeatureContract | FeatureFormat,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
) -> Callable[[type[Any]], type[Any]]:
    """Decorator: register a cell-line featurizer.

    :param name: Registry name used in ``ModelConfig`` and recipes.
    :param description: Short human-readable summary for catalog listings.
    :param contract: Feature format contract for predictor compatibility checks.
    :param tags: Optional discovery tags (for example ``"omics"``).
    :param reference: Optional literature citation metadata.
    :returns: Class decorator that registers the decorated featurizer under *name*.
    """
    return cell_line_featurizer_registry.register(
        name,
        description=description,
        contract=contract,
        tags=tags,
        reference=reference,
    )


def get(name: str) -> type[Any]:
    """Return the cell-line featurizer class registered under *name*.

    :param name: Registry name of the featurizer.
    :returns: Featurizer class registered under *name*.
    :raises ValueError: If *name* is not registered.
    """
    return cell_line_featurizer_registry.get(name)


def list() -> list[str]:  # noqa: A001
    """Return sorted list of registered cell-line featurizer names."""
    return cell_line_featurizer_registry.list_names()


def table() -> pd.DataFrame:
    """Return registry contents as a DataFrame."""
    return cell_line_featurizer_registry.to_dataframe()


def metadata(name: str) -> dict[str, Any]:
    """Return metadata for a registered cell-line featurizer.

    :param name: Registry name of the featurizer.
    :returns: Metadata dict including output format and tags.
    """
    return cell_line_featurizer_registry.get_metadata(name)
