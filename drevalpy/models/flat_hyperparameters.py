"""Compatibility helpers for public hyperparameter view overrides and aliases."""

from __future__ import annotations

from typing import Any

from drevalpy.models.config import ModelConfig
from drevalpy.models.featurizer_mapping import (
    cell_line_featurizer_from_views,
    cell_line_views_from_model_config,
    drug_featurizer_from_view,
    drug_views_from_model_config,
)

# Public compatibility shims: old flat experiment keys -> component-local featurizer keys.
LEGACY_FEATURIZER_FLAT_KEYS: dict[tuple[str, str], dict[str, str]] = {
    ("cell_line", "normalizedProteomics"): {
        "feature_threshold": "proteomics_feature_threshold",
        "n_features": "proteomics_n_features",
        "normalization_width": "proteomics_normalization_width",
        "normalization_downshift": "proteomics_normalization_downshift",
    },
}

PUBLIC_VIEW_KEYS = frozenset({"cell_line_views", "drug_views"})


def _view_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    msg = f"view override must be a string or list, got {type(value).__name__}"
    raise ValueError(msg)


def _warn_legacy_view_keys(flat: dict[str, Any]) -> None:
    present = sorted(PUBLIC_VIEW_KEYS & flat.keys())
    if not present:
        return
    from drevalpy._deprecations import warn_deprecated

    warn_deprecated(
        what="Legacy cell_line_views/drug_views flat hyperparameter API",
        replacement=(
            "explicit cell_line_featurizer/drug_featurizer blocks, recipe strings "
            "(e.g. raw[view]:fingerprints:randomForest), or dotted HPO keys"
        ),
        stacklevel=4,
    )


def _apply_view_overrides(config: ModelConfig, flat: dict[str, Any]) -> ModelConfig:
    """Rewrite featurizers when legacy view lists change.

    :param config: Base model configuration.
    :param flat: Flat hyperparameters that may include legacy view keys.
    :returns: Updated config when view overrides differ from the base config.
    """
    updates: dict[str, Any] = {}
    if "cell_line_views" in flat:
        views = _view_list(flat["cell_line_views"])
        if views != cell_line_views_from_model_config(config):
            updates["cell_line_featurizer"] = cell_line_featurizer_from_views(views, flat)
    if "drug_views" in flat:
        views = _view_list(flat["drug_views"])
        if views != drug_views_from_model_config(config):
            updates["drug_featurizer"] = drug_featurizer_from_view(views[0]) if views else None
    if not updates:
        return config
    return config.model_copy(update=updates, deep=True)


def apply_public_flat_hyperparameters(
    config: ModelConfig,
    flat: dict[str, Any],
    *,
    reject_unknown: bool = True,
    warn_legacy_view_keys: bool = True,
) -> ModelConfig:
    """Backward-compatible wrapper around the collision-aware resolver.

    :param config: Base model configuration.
    :param flat: Flat public hyperparameters to apply.
    :param reject_unknown: Ignored; retained for API compatibility.
    :param warn_legacy_view_keys: Emit deprecation warnings for legacy view keys.
    :returns: Config with public flat hyperparameters applied.
    """
    del reject_unknown
    from drevalpy.components.tuning.public_flat import apply_public_hyperparameters_to_config

    return apply_public_hyperparameters_to_config(
        config,
        flat,
        warn_legacy_view_keys=warn_legacy_view_keys,
    )
