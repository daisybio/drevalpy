"""Translate between public hyperparameter mappings and resolved configs."""

from __future__ import annotations

from typing import Any

from drevalpy.models.config import ModelConfig
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.models.tuning.hyperparameter_export import (
    export_public_mapping,
    export_public_mapping_from_resolved,
)
from drevalpy.models.tuning.hyperparameter_keys import (
    build_ownership_index,
    resolve_to_qualified_mapping,
)

from .search_space import resolve_model_config


def apply_public_hyperparameters_to_config(
    config: ModelConfig,
    mapping: dict[str, Any],
) -> ResolvedModelConfig:
    """Apply a collision-aware public hyperparameter mapping onto a template.

    :param config: Immutable ``ModelConfig`` template.
    :param mapping: Public flat hyperparameter mapping.
    :returns: Resolved instance configuration.
    """
    if not mapping:
        return resolve_model_config(config)

    index = build_ownership_index(config)
    qualified = resolve_to_qualified_mapping(
        config,
        mapping,
        index,
        reserved_keys=frozenset(),
    )
    return resolve_model_config(config, qualified)


def public_hyperparameters_from_config(
    config: ModelConfig | ResolvedModelConfig,
) -> dict[str, Any]:
    """Export a model config into a collision-aware public hyperparameter mapping.

    :param config: Template or resolved configuration.
    :returns: Public hyperparameter mapping; short keys where unambiguous, qualified keys otherwise.
    """
    if isinstance(config, ResolvedModelConfig):
        return export_public_mapping_from_resolved(config)
    return export_public_mapping(config)
