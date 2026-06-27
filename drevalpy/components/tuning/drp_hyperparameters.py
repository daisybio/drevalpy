"""Resolve structured hyperparameter spaces and defaults for DRPModel classes."""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.register_builtins import ensure_components_registered
from drevalpy.models.config import FeaturizerConfig, ModelConfig
from drevalpy.models.featurizer_mapping import cell_line_views_from_model_config, drug_views_from_model_config

from .search_space import (
    apply_merged_to_model_config,
    defaults_from_merged_space,
    merge_model_config_spaces,
)

_LEGACY_FEATURIZER_FLAT_KEYS: dict[tuple[str, str], dict[str, str]] = {
    ("cell_line", "methylationPCA"): {"n_components": "methylation_n_components"},
    ("cell_line", "proteomics"): {
        "feature_threshold": "proteomics_feature_threshold",
        "n_features": "proteomics_n_features",
        "normalization_width": "proteomics_normalization_width",
        "normalization_downshift": "proteomics_normalization_downshift",
    },
}


def model_config_for_drp_model(
    model_class: type[Any],
    hyperparameters: dict[str, Any] | None = None,
) -> ModelConfig | None:
    """Resolve a modular config for a public DRPModel class when possible."""
    ensure_components_registered()
    spec = getattr(model_class, "_model_spec", None)
    if isinstance(spec, str):
        return ModelConfig.from_spec(spec, hyperparameters=hyperparameters)

    model_name = model_class.get_model_name()
    from drevalpy.models.factory import model_config_for_name

    try:
        return model_config_for_name(model_name, hyperparameters or {})
    except KeyError:
        return None


def structured_space_for_drp_model(model_class: type[Any]) -> dict[str, Any]:
    """Return the merged structured search space for a DRPModel class."""
    config = model_config_for_drp_model(model_class)
    if config is None:
        return {}
    return merge_model_config_spaces(config)


def default_hyperparameters_for_drp_model(model_class: type[Any]) -> dict[str, Any]:
    """Return default hyperparameters suitable for ``build_model``."""
    config = model_config_for_drp_model(model_class)
    if config is None:
        return {}
    space = merge_model_config_spaces(config)
    merged_defaults = defaults_from_merged_space(space)
    updated = apply_merged_to_model_config(config, merged_defaults)
    return flat_hyperparameters_from_model_config(updated)


def flat_hyperparameters_from_model_config(config: ModelConfig) -> dict[str, Any]:
    """Flatten a model config into legacy ``build_model`` hyperparameters."""
    flat: dict[str, Any] = {}
    cell_line_views = cell_line_views_from_model_config(config)
    drug_views = drug_views_from_model_config(config)
    if cell_line_views:
        flat["cell_line_views"] = cell_line_views
    if drug_views:
        flat["drug_views"] = drug_views
    if config.predictor is not None:
        from drevalpy.components.registry import get_predictor

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


def config_from_build_hyperparameters(
    model_class: type[Any],
    hyperparameters: dict[str, Any] | None,
) -> ModelConfig | None:
    """Resolve a model config from ``build_model`` hyperparameters."""
    config = model_config_for_drp_model(model_class, hyperparameters)
    if config is None or not hyperparameters:
        return config
    if any("." in key for key in hyperparameters):
        return apply_merged_to_model_config(config, hyperparameters)
    updated = apply_merged_to_model_config(config, hyperparameters)
    return _apply_flat_featurizer_overrides(updated, hyperparameters)


def tuned_flat_hyperparameters(
    model_class: type[Any],
    merged_sample: dict[str, Any],
) -> dict[str, Any]:
    """Convert a merged Ray/Optuna sample into ``build_model`` hyperparameters."""
    config = model_config_for_drp_model(model_class)
    if config is None:
        return dict(merged_sample)
    updated = apply_merged_to_model_config(config, merged_sample)
    return flat_hyperparameters_from_model_config(updated)


def has_tunable_hyperparameters(model_class: type[Any]) -> bool:
    """Return whether the model exposes a non-empty structured search space."""
    return bool(structured_space_for_drp_model(model_class))
