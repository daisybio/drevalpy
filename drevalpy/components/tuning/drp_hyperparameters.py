"""Resolve structured hyperparameter spaces and defaults for DRPModel classes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.register_builtins import ensure_components_registered
from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer, get_predictor
from drevalpy.models.config import FeaturizerConfig, ModelConfig
from drevalpy.models.featurizer_mapping import cell_line_views_from_model_config, drug_views_from_model_config

from .search_space import (
    apply_merged_to_model_config,
    defaults_from_merged_space,
    merge_model_config_spaces,
)

# Public compatibility shims: old flat experiment keys -> component-local featurizer keys.
_LEGACY_FEATURIZER_FLAT_KEYS: dict[tuple[str, str], dict[str, str]] = {
    ("cell_line", "methylationPCA"): {"n_components": "methylation_n_components"},
    ("cell_line", "proteomics"): {
        "feature_threshold": "proteomics_feature_threshold",
        "n_features": "proteomics_n_features",
        "normalization_width": "proteomics_normalization_width",
        "normalization_downshift": "proteomics_normalization_downshift",
    },
}

_PUBLIC_VIEW_KEYS = frozenset({"cell_line_views", "drug_views"})


def base_model_config_for_drp_model(model_class: type[Any]) -> ModelConfig | None:
    """Resolve the base modular config for a public DRPModel class without hyperparameters."""
    ensure_components_registered()
    spec = getattr(model_class, "_model_spec", None)
    if isinstance(spec, str):
        return ModelConfig.from_spec(spec)

    model_name = model_class.get_model_name()
    from drevalpy.models.factory import model_config_for_name

    try:
        return model_config_for_name(model_name, {})
    except KeyError:
        return None


def default_config_for_drp_model(model_class: type[Any]) -> ModelConfig | None:
    """Return a ``ModelConfig`` with structured defaults applied."""
    config = base_model_config_for_drp_model(model_class)
    if config is None:
        return None
    space = merge_model_config_spaces(config)
    merged_defaults = defaults_from_merged_space(space)
    return apply_merged_to_model_config(config, merged_defaults)


def model_config_for_drp_model(
    model_class: type[Any],
    hyperparameters: dict[str, Any] | None = None,
) -> ModelConfig | None:
    """Resolve a modular config for a public DRPModel class."""
    if hyperparameters:
        return config_from_public_hyperparameters(model_class, hyperparameters)
    return base_model_config_for_drp_model(model_class)


def config_from_public_hyperparameters(
    model_class: type[Any],
    hyperparameters: dict[str, Any] | None,
) -> ModelConfig | None:
    """Convert public flat or structured hyperparameters into a ``ModelConfig``."""
    config = base_model_config_for_drp_model(model_class)
    if config is None:
        return None
    if not hyperparameters:
        return config
    if any("." in key for key in hyperparameters):
        return apply_merged_to_model_config(config.model_copy(deep=True), hyperparameters)
    return _apply_public_flat_hyperparameters(config.model_copy(deep=True), hyperparameters)


def public_hyperparameters_from_config(config: ModelConfig) -> dict[str, Any]:
    """Flatten a model config into legacy public ``build_model`` hyperparameters."""
    flat: dict[str, Any] = {}
    cell_line_views = cell_line_views_from_model_config(config)
    drug_views = drug_views_from_model_config(config)
    if cell_line_views:
        flat["cell_line_views"] = cell_line_views
    if drug_views:
        flat["drug_views"] = drug_views
    if config.predictor is not None:
        predictor_cls = get_predictor(config.predictor.name)
        engine_cls = getattr(predictor_cls, "_engine_cls", None)
        if engine_cls is not None:
            flat.update(engine_cls.get_default_hyperparameters())
        else:
            flat.update(predictor_cls.get_default_hyperparameters())
        flat.update(config.predictor.hyperparameters)
    _append_featurizer_flat_keys(flat, config.cell_line_featurizer, "cell_line")
    _append_featurizer_flat_keys(flat, config.drug_featurizer, "drug")
    return flat


def tuned_config_for_drp_model(
    model_class: type[Any],
    merged_sample: dict[str, Any],
) -> ModelConfig | None:
    """Apply a structured Ray/Optuna sample onto the base model config."""
    config = base_model_config_for_drp_model(model_class)
    if config is None:
        return None
    return apply_merged_to_model_config(config, merged_sample)


def build_drp_model_from_config(model: Any, config: ModelConfig) -> None:
    """Build a public DRPModel instance from a resolved ``ModelConfig``."""
    build_from_config = getattr(model, "build_from_model_config", None)
    if callable(build_from_config):
        build_from_config(config)
        return
    model.build_model(public_hyperparameters_from_config(config))


def structured_space_for_drp_model(model_class: type[Any]) -> dict[str, Any]:
    """Return the merged structured search space for a DRPModel class."""
    config = base_model_config_for_drp_model(model_class)
    if config is None:
        return {}
    return merge_model_config_spaces(config)


def default_hyperparameters_for_drp_model(model_class: type[Any]) -> dict[str, Any]:
    """Return default hyperparameters suitable for ``build_model``."""
    config = default_config_for_drp_model(model_class)
    if config is None:
        return {}
    return public_hyperparameters_from_config(config)


def flat_hyperparameters_from_model_config(config: ModelConfig) -> dict[str, Any]:
    """Backward-compatible alias for ``public_hyperparameters_from_config``."""
    return public_hyperparameters_from_config(config)


def config_from_build_hyperparameters(
    model_class: type[Any],
    hyperparameters: dict[str, Any] | None,
) -> ModelConfig | None:
    """Backward-compatible alias for ``config_from_public_hyperparameters``."""
    return config_from_public_hyperparameters(model_class, hyperparameters)


def tuned_flat_hyperparameters(
    model_class: type[Any],
    merged_sample: dict[str, Any],
) -> dict[str, Any]:
    """Convert a merged Ray/Optuna sample into public ``build_model`` hyperparameters."""
    config = tuned_config_for_drp_model(model_class, merged_sample)
    if config is None:
        return dict(merged_sample)
    return public_hyperparameters_from_config(config)


def has_tunable_hyperparameters(model_class: type[Any]) -> bool:
    """Return whether the model exposes a non-empty structured search space."""
    return bool(structured_space_for_drp_model(model_class))


def assert_component_local_hyperparameters(config: ModelConfig) -> None:
    """Raise if namespaced keys leaked into component-local hyperparameter dicts."""
    for featurizer in _iter_featurizer_configs(config):
        for key in featurizer.hyperparameters:
            if key == "featurizers":
                continue
            if "." in key or key.startswith(("featurizer.", "predictor.")):
                msg = f"namespaced key {key!r} found in {featurizer.name} hyperparameters"
                raise AssertionError(msg)
    for key in config.predictor.hyperparameters:
        if key.startswith(("featurizer.", "predictor.")) or key.count(".") >= 2:
            msg = f"namespaced key {key!r} found in predictor hyperparameters"
            raise AssertionError(msg)


def _iter_featurizer_configs(config: ModelConfig) -> Iterator[FeaturizerConfig]:
    for featurizer in (config.cell_line_featurizer, config.drug_featurizer):
        if featurizer is None:
            continue
        yield from _walk_featurizer_configs(featurizer, str(featurizer.registry))


def _walk_featurizer_configs(
    featurizer: FeaturizerConfig,
    registry: str,
) -> Iterator[FeaturizerConfig]:
    if featurizer.name == "concatFeaturizers":
        for child in featurizer.hyperparameters.get("featurizers", []):
            child_cfg = FeaturizerConfig.model_validate(
                normalize_featurizer_config(child, default_registry=registry),
            )
            yield from _walk_featurizer_configs(child_cfg, registry)
        return
    yield featurizer


def _append_featurizer_flat_keys(
    flat: dict[str, Any],
    featurizer: FeaturizerConfig | None,
    registry: str,
) -> None:
    if featurizer is None:
        return
    if featurizer.name == "concatFeaturizers":
        for child in featurizer.hyperparameters.get("featurizers", []):
            child_cfg = FeaturizerConfig.model_validate(
                normalize_featurizer_config(child, default_registry=registry),
            )
            _append_featurizer_flat_keys(flat, child_cfg, registry)
        return
    mapping = _LEGACY_FEATURIZER_FLAT_KEYS.get((registry, featurizer.name), {})
    for component_key, flat_key in mapping.items():
        if component_key in featurizer.hyperparameters:
            flat[flat_key] = featurizer.hyperparameters[component_key]
            if flat_key == "methylation_n_components":
                flat.setdefault("methylation_pca_components", featurizer.hyperparameters[component_key])
    for key, value in featurizer.hyperparameters.items():
        if key == "featurizers":
            continue
        if "." in key or key.startswith(("featurizer.", "predictor.")):
            continue
        if (registry, featurizer.name) not in _LEGACY_FEATURIZER_FLAT_KEYS and key not in flat:
            flat.setdefault(key, value)


def _apply_public_flat_hyperparameters(config: ModelConfig, flat: dict[str, Any]) -> ModelConfig:
    result = _apply_flat_featurizer_overrides(config, flat)
    result = _apply_featurizer_component_flat_keys(result, flat)
    predictor_hp = _extract_predictor_flat_keys(result, flat)
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
    return result


def _apply_flat_featurizer_overrides(config: ModelConfig, flat: dict[str, Any]) -> ModelConfig:
    """Map legacy flat featurizer keys onto a model config."""
    result = config.model_copy(deep=True)
    reverse_map = {
        flat_key: (registry, featurizer_name, component_key)
        for (registry, featurizer_name), mapping in _LEGACY_FEATURIZER_FLAT_KEYS.items()
        for component_key, flat_key in mapping.items()
    }
    for flat_key, value in flat.items():
        mapping = reverse_map.get(flat_key)
        if mapping is None:
            continue
        registry, featurizer_name, component_key = mapping
        target = result.cell_line_featurizer if registry == "cell_line" else result.drug_featurizer
        if target is None:
            continue
        if target.name == featurizer_name:
            target = target.model_copy(
                update={"hyperparameters": {**target.hyperparameters, component_key: value}},
                deep=True,
            )
        elif target.name == "concatFeaturizers":
            children = []
            for child in target.hyperparameters.get("featurizers", []):
                child_cfg = FeaturizerConfig.model_validate(
                    normalize_featurizer_config(child, default_registry=registry),
                )
                if child_cfg.name == featurizer_name:
                    child_cfg = child_cfg.model_copy(
                        update={"hyperparameters": {**child_cfg.hyperparameters, component_key: value}},
                        deep=True,
                    )
                children.append(child_cfg.model_dump())
            target = target.model_copy(
                update={"hyperparameters": {**target.hyperparameters, "featurizers": children}},
                deep=True,
            )
        if registry == "cell_line":
            result.cell_line_featurizer = target
        else:
            result.drug_featurizer = target
    return result


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
    if featurizer.name == "concatFeaturizers":
        children = []
        for child in featurizer.hyperparameters.get("featurizers", []):
            child_cfg = FeaturizerConfig.model_validate(
                normalize_featurizer_config(child, default_registry=registry),
            )
            children.append(_apply_flat_keys_to_featurizer_tree(child_cfg, flat, registry).model_dump())
        return featurizer.model_copy(
            update={"hyperparameters": {**featurizer.hyperparameters, "featurizers": children}},
            deep=True,
        )

    space_keys = set(_featurizer_cls(featurizer, registry).get_hyperparameter_space())
    updates = {key: flat[key] for key in space_keys if key in flat}
    if not updates:
        return featurizer
    return featurizer.model_copy(
        update={"hyperparameters": {**featurizer.hyperparameters, **updates}},
        deep=True,
    )


def _featurizer_cls(featurizer: FeaturizerConfig, registry: str) -> type[Any]:
    if registry == "cell_line":
        return get_cell_line_featurizer(featurizer.name)
    return get_drug_featurizer(featurizer.name)


def _extract_predictor_flat_keys(config: ModelConfig, flat: dict[str, Any]) -> dict[str, Any]:
    predictor_cls = get_predictor(config.predictor.name)
    engine_cls = getattr(predictor_cls, "_engine_cls", None)
    if engine_cls is not None:
        predictor_keys = set(engine_cls.get_default_hyperparameters())
    else:
        predictor_keys = set(predictor_cls.get_default_hyperparameters())
    reserved = _PUBLIC_VIEW_KEYS | _featurizer_public_flat_keys(config)
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
            legacy = _LEGACY_FEATURIZER_FLAT_KEYS.get((registry_name, child.name), {})
            keys.update(legacy.values())
    return keys
