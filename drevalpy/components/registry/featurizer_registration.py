"""Public register / get / list helpers for featurizer registries."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from drevalpy.components.core.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.core.plugins.register_builtins import register_builtin_components
from drevalpy.components.registry.featurizer_registry import (
    cell_line_featurizer_registry,
    drug_featurizer_registry,
)
from drevalpy.types.enums.literature_reference import LiteratureReference


def register_cell_line_featurizer(
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
        tags=tags,
        reference=reference,
        contract=contract,
    )


def get_cell_line_featurizer(name: str) -> type[Any]:
    """Return the cell-line featurizer class registered under *name*.

    :param name: Registry name of the featurizer.

    :returns: Featurizer class registered under *name*.

    :raises ValueError: If *name* is not a known built-in featurizer.
    :raises ImportError: If the featurizer's optional dependency is unavailable.
    """
    if name not in cell_line_featurizer_registry.list_names():
        from drevalpy.components.core.plugins.register_builtins import (
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
    """List all registered cell-line featurizer names.

    :returns: Registry names after ensuring built-in components are loaded.
    """
    register_builtin_components()
    return cell_line_featurizer_registry.list_names()


def get_cell_line_featurizer_metadata(name: str) -> dict[str, Any]:
    """Return metadata for a registered cell-line featurizer.

    :param name: Registry name of the featurizer.

    :returns: Metadata dict including output format and tags.
    """
    get_cell_line_featurizer(name)
    return cell_line_featurizer_registry.get_metadata(name)


def list_cell_line_featurizer_metadata(*, tag: str | None = None) -> list[dict[str, Any]]:
    """List metadata for all registered cell-line featurizers.

    :param tag: When set, keep only featurizers whose ``tags`` contain *tag*.

    :returns: List of metadata dicts.
    """
    register_builtin_components()
    return cell_line_featurizer_registry.list_metadata(tag=tag)


def register_drug_featurizer(
    name: str,
    *,
    description: str,
    contract: FeatureContract | FeatureFormat,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
) -> Callable[[type[Any]], type[Any]]:
    """Decorator: register a drug featurizer.

    :param name: Registry name used in ``ModelConfig`` and recipes.
    :param description: Short human-readable summary for catalog listings.
    :param contract: Feature format contract for predictor compatibility checks.
    :param tags: Optional discovery tags.
    :param reference: Optional literature citation metadata.

    :returns: Class decorator that registers the decorated featurizer under *name*.
    """
    return drug_featurizer_registry.register(
        name,
        description=description,
        tags=tags,
        reference=reference,
        contract=contract,
    )


def get_drug_featurizer(name: str) -> type[Any]:
    """Return the drug featurizer class registered under *name*.

    :param name: Registry name of the featurizer.

    :returns: Featurizer class registered under *name*.

    :raises ValueError: If *name* is not a known built-in featurizer.
    :raises ImportError: If the featurizer's optional dependency is unavailable.
    """
    if name not in drug_featurizer_registry.list_names():
        from drevalpy.components.core.plugins.register_builtins import (
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
    """List all registered drug featurizer names.

    :returns: Registry names after ensuring built-in components are loaded.
    """
    register_builtin_components()
    return drug_featurizer_registry.list_names()


def get_drug_featurizer_metadata(name: str) -> dict[str, Any]:
    """Return metadata for a registered drug featurizer.

    :param name: Registry name of the featurizer.

    :returns: Metadata dict including output format and tags.
    """
    get_drug_featurizer(name)
    return drug_featurizer_registry.get_metadata(name)


def list_drug_featurizer_metadata(*, tag: str | None = None) -> list[dict[str, Any]]:
    """List metadata for all registered drug featurizers.

    :param tag: When set, keep only featurizers whose ``tags`` contain *tag*.

    :returns: List of metadata dicts.
    """
    register_builtin_components()
    return drug_featurizer_registry.list_metadata(tag=tag)
