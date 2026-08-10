"""Public register / get / list helpers for the predictor registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from drevalpy.components.core.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.core.plugins.register_builtins import register_builtin_components
from drevalpy.components.registry.predictor_registry import predictor_registry
from drevalpy.types.enums.literature_reference import LiteratureReference


def register_predictor(
    name: str,
    *,
    description: str,
    cell_line_contract: FeatureContract | FeatureFormat,
    drug_contract: FeatureContract | FeatureFormat,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
) -> Callable[[type[Any]], type[Any]]:
    """Decorator: register a predictor.

    :param name: Registry name used in ``ModelConfig`` and recipes.
    :param description: Short human-readable summary for catalog listings.
    :param cell_line_contract: Expected cell-line feature format.
    :param drug_contract: Expected drug feature format.
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


def get_predictor(name: str) -> type[Any]:
    """Return the predictor class registered under *name*.

    :param name: Registry name of the predictor.

    :returns: Predictor class registered under *name*.

    :raises ValueError: If *name* is not a known built-in predictor.
    :raises ImportError: If the predictor's optional dependency is unavailable.
    """
    if name not in predictor_registry.list_names():
        from drevalpy.components.core.plugins.register_builtins import (
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
    """List all registered predictor names.

    :returns: Registry names after ensuring built-in components are loaded.
    """
    register_builtin_components()
    return predictor_registry.list_names()


def get_predictor_metadata(name: str) -> dict[str, Any]:
    """Return metadata for a registered predictor.

    :param name: Registry name of the predictor.

    :returns: Metadata dict including input interface, tags, and literature fields.
    """
    get_predictor(name)
    return predictor_registry.get_metadata(name)


def list_predictor_metadata(*, tag: str | None = None) -> list[dict[str, Any]]:
    """List metadata for all registered predictors.

    :param tag: When set, keep only predictors whose ``tags`` contain *tag*.

    :returns: List of metadata dicts.
    """
    register_builtin_components()
    return predictor_registry.list_metadata(tag=tag)
