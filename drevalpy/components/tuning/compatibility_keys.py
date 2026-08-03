"""Legacy flat-key aliases and config-to-public featurizer translation."""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer
from drevalpy.models.config import FeaturizerConfig
from drevalpy.models.flat_hyperparameters import (
    LEGACY_FEATURIZER_FLAT_KEYS,
    PUBLIC_VIEW_KEYS,
)

# Backward-compatible aliases for callers/tests that import private names.
_LEGACY_FEATURIZER_FLAT_KEYS = LEGACY_FEATURIZER_FLAT_KEYS
_PUBLIC_VIEW_KEYS = PUBLIC_VIEW_KEYS


def _featurizer_space_keys(featurizer: FeaturizerConfig, registry: str) -> set[str]:
    cls = get_cell_line_featurizer(featurizer.name) if registry == "cell_line" else get_drug_featurizer(featurizer.name)
    return set(cls.get_hyperparameter_space())


def _is_exportable_space_flat_key(registry: str, featurizer_name: str, key: str, flat: dict[str, Any]) -> bool:
    if key in {"featurizers", "view", "views"}:
        return False
    if "." in key or key.startswith(("cell_line_featurizer.", "drug_featurizer.", "predictor.")):
        return False
    if (registry, featurizer_name) in _LEGACY_FEATURIZER_FLAT_KEYS:
        return False
    return key not in flat


def _append_legacy_component_flat_keys(
    flat: dict[str, Any],
    featurizer: FeaturizerConfig,
    registry: str,
) -> None:
    mapping = _LEGACY_FEATURIZER_FLAT_KEYS.get((registry, featurizer.name), {})
    for component_key, flat_key in mapping.items():
        if component_key in featurizer.hyperparameters:
            flat[flat_key] = featurizer.hyperparameters[component_key]


def _append_methylation_pca_flat_aliases(flat: dict[str, Any], featurizer: FeaturizerConfig) -> None:
    if featurizer.name == "pca" and featurizer.view == "methylation" and "n_components" in featurizer.hyperparameters:
        flat["methylation_n_components"] = featurizer.hyperparameters["n_components"]
        flat.setdefault("methylation_pca_components", featurizer.hyperparameters["n_components"])


def _append_hyperparameter_space_flat_keys(
    flat: dict[str, Any],
    featurizer: FeaturizerConfig,
    registry: str,
) -> None:
    space_keys = _featurizer_space_keys(featurizer, registry)
    for key, value in featurizer.hyperparameters.items():
        if key not in space_keys:
            continue
        if not _is_exportable_space_flat_key(registry, featurizer.name, key, flat):
            continue
        flat.setdefault(key, value)


def append_featurizer_flat_keys(
    flat: dict[str, Any],
    featurizer: FeaturizerConfig | None,
    registry: str,
) -> None:
    """Append legacy and tunable featurizer keys into a public flat dict.

    Architecture-only featurizer kwargs (present in ModelConfig but absent from
    ``get_hyperparameter_space``) stay on the config tree and are not flattened.
    """
    if featurizer is None:
        return
    if featurizer.name == "concatFeaturizers":
        for child in featurizer.hyperparameters.get("featurizers", []):
            child_cfg = FeaturizerConfig.model_validate(
                normalize_featurizer_config(child, default_registry=registry),
            )
            append_featurizer_flat_keys(flat, child_cfg, registry)
        return
    _append_legacy_component_flat_keys(flat, featurizer, registry)
    if registry == "cell_line":
        _append_methylation_pca_flat_aliases(flat, featurizer)
    _append_hyperparameter_space_flat_keys(flat, featurizer, registry)
