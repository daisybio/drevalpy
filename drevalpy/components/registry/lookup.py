"""Register, resolve, and clear model components in the global registries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from drevalpy.components.registry.core import (
    cell_line_featurizer_registry,
    drug_featurizer_registry,
    predictor_registry,
)


def register_cell_line_featurizer(
    name: str,
    *,
    description: str,
    category: str,
    template_repo_url: str = "",
    citation: str = "",
    citation_doi: str = "",
    citation_text: str = "",
    deviations: str = "",
) -> Callable[[type[Any]], type[Any]]:
    """Decorator: register a cell-line featurizer."""
    return cell_line_featurizer_registry.register(
        name,
        description=description,
        category=category,
        template_repo_url=template_repo_url,
        citation=citation,
        citation_doi=citation_doi,
        citation_text=citation_text,
        deviations=deviations,
    )


def get_cell_line_featurizer(name: str) -> type[Any]:
    """Return the cell-line featurizer class registered under *name*."""
    return cell_line_featurizer_registry.get(name)


def list_cell_line_featurizers() -> list[str]:
    """List all registered cell-line featurizer names."""
    return cell_line_featurizer_registry.list_names()


def get_cell_line_featurizer_metadata(name: str) -> dict[str, str]:
    """Return metadata for a registered cell-line featurizer."""
    return cell_line_featurizer_registry.get_metadata(name)


def list_cell_line_featurizer_metadata(category: str | None = None) -> list[dict[str, str]]:
    """List metadata for all registered cell-line featurizers."""
    return cell_line_featurizer_registry.list_metadata(category)


def clear_cell_line_featurizer_registry() -> None:
    """Clear the cell-line featurizer registry (primarily for testing)."""
    cell_line_featurizer_registry.clear()


def register_drug_featurizer(
    name: str,
    *,
    description: str,
    category: str,
    template_repo_url: str = "",
    citation: str = "",
    citation_doi: str = "",
    citation_text: str = "",
    deviations: str = "",
) -> Callable[[type[Any]], type[Any]]:
    """Decorator: register a drug featurizer."""
    return drug_featurizer_registry.register(
        name,
        description=description,
        category=category,
        template_repo_url=template_repo_url,
        citation=citation,
        citation_doi=citation_doi,
        citation_text=citation_text,
        deviations=deviations,
    )


def get_drug_featurizer(name: str) -> type[Any]:
    """Return the drug featurizer class registered under *name*."""
    return drug_featurizer_registry.get(name)


def list_drug_featurizers() -> list[str]:
    """List all registered drug featurizer names."""
    return drug_featurizer_registry.list_names()


def get_drug_featurizer_metadata(name: str) -> dict[str, str]:
    """Return metadata for a registered drug featurizer."""
    return drug_featurizer_registry.get_metadata(name)


def list_drug_featurizer_metadata(category: str | None = None) -> list[dict[str, str]]:
    """List metadata for all registered drug featurizers."""
    return drug_featurizer_registry.list_metadata(category)


def clear_drug_featurizer_registry() -> None:
    """Clear the drug featurizer registry (primarily for testing)."""
    drug_featurizer_registry.clear()


def register_predictor(
    name: str,
    *,
    description: str,
    category: str,
    template_repo_url: str = "",
    citation: str = "",
    citation_doi: str = "",
    citation_text: str = "",
    deviations: str = "",
) -> Callable[[type[Any]], type[Any]]:
    """Decorator: register a predictor."""
    return predictor_registry.register(
        name,
        description=description,
        category=category,
        template_repo_url=template_repo_url,
        citation=citation,
        citation_doi=citation_doi,
        citation_text=citation_text,
        deviations=deviations,
    )


def get_predictor(name: str) -> type[Any]:
    """Return the predictor class registered under *name*."""
    return predictor_registry.get(name)


def list_predictors() -> list[str]:
    """List all registered predictor names."""
    return predictor_registry.list_names()


def get_predictor_metadata(name: str) -> dict[str, str]:
    """Return metadata for a registered predictor."""
    return predictor_registry.get_metadata(name)


def list_predictor_metadata(category: str | None = None) -> list[dict[str, str]]:
    """List metadata for all registered predictors."""
    return predictor_registry.list_metadata(category)


def clear_predictor_registry() -> None:
    """Clear the predictor registry (primarily for testing)."""
    predictor_registry.clear()
