"""Translate between public hyperparameter mappings and resolved configs."""

from __future__ import annotations

from typing import Any

from drevalpy.models.config import ModelConfig
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.models.tuning.compatibility_keys import PUBLIC_VIEW_KEYS
from drevalpy.models.tuning.hyperparameter_export import (
    export_public_mapping,
    export_public_mapping_from_resolved,
)
from drevalpy.models.tuning.hyperparameter_keys import (
    build_ownership_index,
    resolve_to_qualified_mapping,
)

from ._model_config_base import base_model_config_for_drp_model
from .search_space import resolve_model_config


def apply_public_hyperparameters_to_config(
    config: ModelConfig,
    mapping: dict[str, Any],
) -> ResolvedModelConfig:
    """Apply a collision-aware public hyperparameter mapping onto a template.

    :param config: Immutable ``ModelConfig`` template.
    :param mapping: Public flat hyperparameter mapping.
    :returns: Resolved instance configuration.
    :raises ValueError: If legacy view keys are present in *mapping*.
    """
    if not mapping:
        return resolve_model_config(config)

    present = sorted(PUBLIC_VIEW_KEYS & mapping.keys())
    if present:
        msg = (
            f"Legacy view keys {present!r} are no longer supported. "
            "Use explicit cell_line_featurizer/drug_featurizer blocks, recipe strings "
            "(e.g. raw[view]:fingerprints:randomForest), or dotted HPO keys instead."
        )
        raise ValueError(msg)

    normalized = dict(mapping)
    if "methylation_n_components" not in normalized and "methylation_pca_components" in normalized:
        normalized["methylation_n_components"] = normalized.pop("methylation_pca_components")

    index = build_ownership_index(config)
    qualified = resolve_to_qualified_mapping(
        config,
        normalized,
        index,
        reserved_keys=frozenset(),
    )
    return resolve_model_config(config, qualified)


def public_hyperparameters_from_config(
    config: ModelConfig | ResolvedModelConfig,
    *,
    include_view_keys: bool = False,
) -> dict[str, Any]:
    """Export a model config into a collision-aware public hyperparameter mapping.

    :param config: Template or resolved configuration.
    :param include_view_keys: include view keys.
    :returns: Result.
    """
    if isinstance(config, ResolvedModelConfig):
        return export_public_mapping_from_resolved(config, include_view_keys=include_view_keys)
    return export_public_mapping(config, include_view_keys=include_view_keys)


def config_from_public_hyperparameters(
    model_class: type[Any],
    hyperparameters: dict[str, Any] | None,
) -> ResolvedModelConfig | None:
    """Convert a public hyperparameter mapping into a resolved config.

    :param model_class: model class.
    :param hyperparameters: hyperparameters.
    :returns: Result.
    """
    config = base_model_config_for_drp_model(model_class)
    if config is None:
        return None
    if not hyperparameters:
        return resolve_model_config(config)
    return apply_public_hyperparameters_to_config(config, hyperparameters)
