"""Register, resolve, and clear model components in the global registries."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.registry.core import (
    cell_line_featurizer_registry,
    drug_featurizer_registry,
    predictor_registry,
)
from drevalpy.types.literature_reference import LiteratureReference


def _ensure_builtins_for_discovery(*, registry_names: list[str]) -> None:
    """Register built-ins only when the target registry is still empty."""
    if registry_names:
        return
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()


def register_cell_line_featurizer(
    name: str,
    *,
    description: str,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
    contract: FeatureContract | FeatureFormat | None = None,
) -> Callable[[type[Any]], type[Any]]:
    """Decorator: register a cell-line featurizer."""
    return cell_line_featurizer_registry.register(
        name,
        description=description,
        tags=tags,
        reference=reference,
        contract=contract,
    )


def get_cell_line_featurizer(name: str) -> type[Any]:
    """Return the cell-line featurizer class registered under *name*."""
    if name not in cell_line_featurizer_registry.list_names():
        from drevalpy.components.register_builtins import (
            ensure_cell_line_featurizer_registered,
            is_known_builtin_cell_line_featurizer,
        )

        if not is_known_builtin_cell_line_featurizer(name):
            raise ValueError(f"Unknown Cell line featurizer: {name!r}")
        try:
            ensure_cell_line_featurizer_registered(name)
        except ImportError:
            raise
        except Exception as exc:
            raise ImportError(f"Cell line featurizer {name!r} could not be imported: {exc}") from exc
    if name not in cell_line_featurizer_registry.list_names():
        raise ImportError(f"Cell line featurizer {name!r} is unavailable; its optional dependency was not registered.")
    return cell_line_featurizer_registry.get(name)


def list_cell_line_featurizers() -> list[str]:
    """List all registered cell-line featurizer names."""
    _ensure_builtins_for_discovery(registry_names=cell_line_featurizer_registry.list_names())
    return cell_line_featurizer_registry.list_names()


def get_cell_line_featurizer_metadata(name: str) -> dict[str, str]:
    """Return metadata for a registered cell-line featurizer."""
    get_cell_line_featurizer(name)
    return cell_line_featurizer_registry.get_metadata(name)


def list_cell_line_featurizer_metadata(*, tag: str | None = None) -> list[dict[str, str]]:
    """List metadata for all registered cell-line featurizers."""
    _ensure_builtins_for_discovery(registry_names=cell_line_featurizer_registry.list_names())
    return cell_line_featurizer_registry.list_metadata(tag=tag)


def clear_cell_line_featurizer_registry() -> None:
    """Clear the cell-line featurizer registry (primarily for testing)."""
    cell_line_featurizer_registry.clear()


def register_drug_featurizer(
    name: str,
    *,
    description: str,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
    contract: FeatureContract | FeatureFormat | None = None,
) -> Callable[[type[Any]], type[Any]]:
    """Decorator: register a drug featurizer."""
    return drug_featurizer_registry.register(
        name,
        description=description,
        tags=tags,
        reference=reference,
        contract=contract,
    )


def get_drug_featurizer(name: str) -> type[Any]:
    """Return the drug featurizer class registered under *name*."""
    if name not in drug_featurizer_registry.list_names():
        from drevalpy.components.register_builtins import (
            ensure_drug_featurizer_registered,
            is_known_builtin_drug_featurizer,
        )

        if not is_known_builtin_drug_featurizer(name):
            raise ValueError(f"Unknown Drug featurizer: {name!r}")
        try:
            ensure_drug_featurizer_registered(name)
        except ImportError:
            raise
        except Exception as exc:
            raise ImportError(f"Drug featurizer {name!r} could not be imported: {exc}") from exc
    if name not in drug_featurizer_registry.list_names():
        raise ImportError(f"Drug featurizer {name!r} is unavailable; its optional dependency was not registered.")
    return drug_featurizer_registry.get(name)


def list_drug_featurizers() -> list[str]:
    """List all registered drug featurizer names."""
    _ensure_builtins_for_discovery(registry_names=drug_featurizer_registry.list_names())
    return drug_featurizer_registry.list_names()


def get_drug_featurizer_metadata(name: str) -> dict[str, str]:
    """Return metadata for a registered drug featurizer."""
    get_drug_featurizer(name)
    return drug_featurizer_registry.get_metadata(name)


def list_drug_featurizer_metadata(*, tag: str | None = None) -> list[dict[str, str]]:
    """List metadata for all registered drug featurizers."""
    _ensure_builtins_for_discovery(registry_names=drug_featurizer_registry.list_names())
    return drug_featurizer_registry.list_metadata(tag=tag)


def clear_drug_featurizer_registry() -> None:
    """Clear the drug featurizer registry (primarily for testing)."""
    drug_featurizer_registry.clear()


def register_predictor(
    name: str,
    *,
    description: str,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
    cell_line_contract: FeatureContract | FeatureFormat | None = None,
    drug_contract: FeatureContract | FeatureFormat | None = None,
) -> Callable[[type[Any]], type[Any]]:
    """Decorator: register a predictor."""
    return predictor_registry.register(
        name,
        description=description,
        tags=tags,
        reference=reference,
        cell_line_contract=cell_line_contract,
        drug_contract=drug_contract,
    )


def get_predictor(name: str) -> type[Any]:
    """Return the predictor class registered under *name*."""
    if name not in predictor_registry.list_names():
        from drevalpy.components.register_builtins import (
            ensure_predictor_registered,
            is_known_builtin_predictor,
        )

        if not is_known_builtin_predictor(name):
            raise ValueError(f"Unknown Predictor: {name!r}")
        try:
            ensure_predictor_registered(name)
        except ImportError:
            raise
        except Exception as exc:
            raise ImportError(f"Predictor {name!r} could not be imported: {exc}") from exc
    if name not in predictor_registry.list_names():
        raise ImportError(f"Predictor {name!r} is unavailable; its optional/literature dependency was not registered.")
    return predictor_registry.get(name)


def list_predictors() -> list[str]:
    """List all registered predictor names."""
    _ensure_builtins_for_discovery(registry_names=predictor_registry.list_names())
    return predictor_registry.list_names()


def get_predictor_metadata(name: str) -> dict[str, str]:
    """Return metadata for a registered predictor."""
    get_predictor(name)
    return predictor_registry.get_metadata(name)


def list_predictor_metadata(*, tag: str | None = None) -> list[dict[str, str]]:
    """List metadata for all registered predictors."""
    _ensure_builtins_for_discovery(registry_names=predictor_registry.list_names())
    return predictor_registry.list_metadata(tag=tag)


def clear_predictor_registry() -> None:
    """Clear the predictor registry (primarily for testing)."""
    predictor_registry.clear()
