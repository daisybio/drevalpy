"""Single-path public flat hyperparameter application for ModelConfig."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from drevalpy.components.featurizer_tree import iter_featurizer_leaves, map_featurizer_tree
from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer, get_predictor
from drevalpy.models.config import FeaturizerConfig, ModelConfig
from drevalpy.models.featurizer_mapping import (
    cell_line_featurizer_from_views,
    cell_line_views_from_model_config,
    drug_featurizer_from_view,
    drug_views_from_model_config,
)

logger = logging.getLogger(__name__)

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
_METHYLATION_FLAT_KEYS = frozenset({"methylation_n_components", "methylation_pca_components"})


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


def apply_public_flat_hyperparameters(
    config: ModelConfig,
    flat: dict[str, Any],
    *,
    reject_unknown: bool = True,
    warn_legacy_view_keys: bool = True,
) -> ModelConfig:
    """Apply public flat hyperparameters onto a ModelConfig.

    Handles view overrides, legacy featurizer keys, predictor keys, and
    methylation PCA aliases. Does not leak view keys into predictor hyperparameters.

    ``cell_line_views`` / ``drug_views`` remain supported for compatibility but
    emit a ``FutureWarning`` when ``warn_legacy_view_keys`` is true.
    """
    if not flat:
        return config.model_copy(deep=True)
    if warn_legacy_view_keys:
        _warn_legacy_view_keys(flat)
    normalized = dict(flat)
    if "methylation_n_components" not in normalized and "methylation_pca_components" in normalized:
        normalized["methylation_n_components"] = normalized["methylation_pca_components"]

    result = _apply_view_overrides(config.model_copy(deep=True), normalized)
    result = _apply_flat_featurizer_overrides(result, normalized)
    result = _apply_featurizer_component_flat_keys(result, normalized)
    predictor_hp = _extract_predictor_flat_keys(result, normalized)
    if predictor_hp:
        result = result.model_copy(
            update={
                "predictor": result.predictor.model_copy(
                    update={"hyperparameters": {**result.predictor.hyperparameters, **predictor_hp}},
                    deep=True,
                )
            },
            deep=True,
        )

    if reject_unknown:
        _reject_unmapped_flat_keys(result, normalized, predictor_hp)
    return result


def _apply_view_overrides(config: ModelConfig, flat: dict[str, Any]) -> ModelConfig:
    """Rewrite featurizers when view lists change.

    If the requested views match the config's current views, keep the existing
    featurizer tree so flat keys such as ``n_components`` still apply (e.g. a
    ``pca[expression]`` recipe that also emits ``cell_line_views``).
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


def _walk_featurizer_configs(
    featurizer: FeaturizerConfig,
    registry: str,
) -> Iterator[FeaturizerConfig]:
    yield from iter_featurizer_leaves(featurizer, registry)


def _legacy_featurizer_flat_reverse_map() -> dict[str, tuple[str, str, str]]:
    reverse_map = {
        flat_key: (registry, featurizer_name, component_key)
        for (registry, featurizer_name), mapping in LEGACY_FEATURIZER_FLAT_KEYS.items()
        for component_key, flat_key in mapping.items()
    }
    reverse_map["methylation_n_components"] = ("cell_line", "pca", "n_components")
    reverse_map["methylation_pca_components"] = ("cell_line", "pca", "n_components")
    return reverse_map


def _patch_featurizer_legacy_key(
    target: FeaturizerConfig,
    *,
    registry: str,
    featurizer_name: str,
    component_key: str,
    value: Any,
) -> FeaturizerConfig:
    if target.name == featurizer_name:
        return target.model_copy(
            update={"hyperparameters": {**target.hyperparameters, component_key: value}},
            deep=True,
        )
    if target.name != "concatFeaturizers":
        return target

    def _patch_leaf(child: FeaturizerConfig) -> FeaturizerConfig:
        if child.name == featurizer_name and (featurizer_name != "pca" or child.view == "methylation"):
            return child.model_copy(
                update={"hyperparameters": {**child.hyperparameters, component_key: value}},
                deep=True,
            )
        return child

    return map_featurizer_tree(target, registry, _patch_leaf)


def _assign_featurizer_on_config(config: ModelConfig, registry: str, featurizer: FeaturizerConfig) -> ModelConfig:
    if registry == "cell_line":
        return config.model_copy(update={"cell_line_featurizer": featurizer}, deep=True)
    return config.model_copy(update={"drug_featurizer": featurizer}, deep=True)


def _apply_flat_featurizer_overrides(config: ModelConfig, flat: dict[str, Any]) -> ModelConfig:
    result = config.model_copy(deep=True)
    reverse_map = _legacy_featurizer_flat_reverse_map()
    for flat_key, value in flat.items():
        if flat_key in _METHYLATION_FLAT_KEYS:
            result = _apply_pca_methylation_flat_key(result, value)
            continue
        mapping = reverse_map.get(flat_key)
        if mapping is None:
            continue
        registry, featurizer_name, component_key = mapping
        target = result.cell_line_featurizer if registry == "cell_line" else result.drug_featurizer
        if target is None:
            continue
        patched = _patch_featurizer_legacy_key(
            target,
            registry=registry,
            featurizer_name=featurizer_name,
            component_key=component_key,
            value=value,
        )
        result = _assign_featurizer_on_config(result, registry, patched)
    return result


def _apply_pca_methylation_flat_key(config: ModelConfig, value: Any) -> ModelConfig:
    target = config.cell_line_featurizer
    if target is None:
        return config
    updated = _set_pca_methylation_n_components(target, value)
    return config.model_copy(update={"cell_line_featurizer": updated}, deep=True)


def _set_pca_methylation_n_components(featurizer: FeaturizerConfig, value: Any) -> FeaturizerConfig:
    if featurizer.name == "pca" and featurizer.view == "methylation":
        return featurizer.model_copy(
            update={"hyperparameters": {**featurizer.hyperparameters, "n_components": value}},
            deep=True,
        )

    def _patch_methylation_pca(child: FeaturizerConfig) -> FeaturizerConfig:
        if child.name == "pca" and child.view == "methylation":
            return child.model_copy(
                update={"hyperparameters": {**child.hyperparameters, "n_components": value}},
                deep=True,
            )
        return child

    return map_featurizer_tree(featurizer, "cell_line", _patch_methylation_pca)


def _apply_featurizer_component_flat_keys(config: ModelConfig, flat: dict[str, Any]) -> ModelConfig:
    result = config.model_copy(deep=True)
    for registry_name, attr in (("cell_line", "cell_line_featurizer"), ("drug", "drug_featurizer")):
        featurizer = getattr(result, attr)
        if featurizer is None:
            continue
        updated = _apply_flat_keys_to_featurizer_tree(featurizer, flat, registry_name)
        setattr(result, attr, updated)
    return result


def _apply_flat_keys_to_featurizer_tree(
    featurizer: FeaturizerConfig,
    flat: dict[str, Any],
    registry: str,
) -> FeaturizerConfig:
    def _apply_leaf(child: FeaturizerConfig) -> FeaturizerConfig:
        space_keys = set(_featurizer_cls(child, registry).get_hyperparameter_space())
        updates = {key: flat[key] for key in space_keys if key in flat}
        if not updates:
            return child
        return child.model_copy(
            update={"hyperparameters": {**child.hyperparameters, **updates}},
            deep=True,
        )

    return map_featurizer_tree(featurizer, registry, _apply_leaf)


def _featurizer_cls(featurizer: FeaturizerConfig, registry: str) -> type[Any]:
    if registry == "cell_line":
        return get_cell_line_featurizer(featurizer.name)
    return get_drug_featurizer(featurizer.name)


def _predictor_accepted_keys(predictor_cls: type[Any]) -> set[str]:
    engine_cls = getattr(predictor_cls, "_engine_cls", None)
    source = engine_cls if engine_cls is not None else predictor_cls
    keys = set(source.get_default_hyperparameters())
    keys.update(source.get_hyperparameter_space())
    non_tunable = getattr(source, "non_tunable_hyperparameters", None)
    if isinstance(non_tunable, dict):
        keys.update(non_tunable)
    elif isinstance(non_tunable, (set, frozenset, list, tuple)):
        keys.update(str(key) for key in non_tunable)
    return keys


def _extract_predictor_flat_keys(config: ModelConfig, flat: dict[str, Any]) -> dict[str, Any]:
    predictor_cls = get_predictor(config.predictor.name)
    predictor_keys = _predictor_accepted_keys(predictor_cls)
    reserved = PUBLIC_VIEW_KEYS | _featurizer_public_flat_keys(config)
    return {
        key: value for key, value in flat.items() if key in predictor_keys and key not in reserved and "." not in key
    }


def _featurizer_public_flat_keys(config: ModelConfig) -> set[str]:
    keys: set[str] = set()
    for registry_name, attr in (("cell_line", "cell_line_featurizer"), ("drug", "drug_featurizer")):
        featurizer = getattr(config, attr)
        if featurizer is None:
            continue
        for child in _walk_featurizer_configs(featurizer, registry_name):
            keys.update(_featurizer_cls(child, registry_name).get_hyperparameter_space())
            legacy = LEGACY_FEATURIZER_FLAT_KEYS.get((registry_name, child.name), {})
            keys.update(legacy.values())
            if registry_name == "cell_line" and child.name == "pca" and child.view == "methylation":
                keys.update(_METHYLATION_FLAT_KEYS)
    return keys


def _reject_unmapped_flat_keys(
    config: ModelConfig,
    flat: dict[str, Any],
    predictor_hp: dict[str, Any],
) -> None:
    reserved = PUBLIC_VIEW_KEYS | _featurizer_public_flat_keys(config) | set(predictor_hp) | _METHYLATION_FLAT_KEYS
    # Featurizer component keys applied via space are already in reserved via _featurizer_public_flat_keys.
    unknown = [key for key in flat if key not in reserved and "." not in key]
    if unknown:
        preview = ", ".join(repr(key) for key in sorted(unknown)[:8])
        suffix = f" (+{len(unknown) - 8} more)" if len(unknown) > 8 else ""
        msg = f"Unknown public hyperparameters for predictor {config.predictor.name!r}: {preview}{suffix}"
        raise ValueError(msg)
