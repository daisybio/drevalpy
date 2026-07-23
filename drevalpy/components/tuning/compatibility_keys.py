"""Legacy flat-key aliases and config-to-public featurizer translation."""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.models.config import FeaturizerConfig
from drevalpy.models.flat_hyperparameters import LEGACY_FEATURIZER_FLAT_KEYS, PUBLIC_VIEW_KEYS

# Backward-compatible aliases for callers/tests that import private names.
_LEGACY_FEATURIZER_FLAT_KEYS = LEGACY_FEATURIZER_FLAT_KEYS
_PUBLIC_VIEW_KEYS = PUBLIC_VIEW_KEYS


def append_featurizer_flat_keys(
    flat: dict[str, Any],
    featurizer: FeaturizerConfig | None,
    registry: str,
) -> None:
    """Append legacy and component-local featurizer keys into a public flat dict."""
    if featurizer is None:
        return
    if featurizer.name == "concatFeaturizers":
        for child in featurizer.hyperparameters.get("featurizers", []):
            child_cfg = FeaturizerConfig.model_validate(
                normalize_featurizer_config(child, default_registry=registry),
            )
            append_featurizer_flat_keys(flat, child_cfg, registry)
        return
    mapping = _LEGACY_FEATURIZER_FLAT_KEYS.get((registry, featurizer.name), {})
    for component_key, flat_key in mapping.items():
        if component_key in featurizer.hyperparameters:
            flat[flat_key] = featurizer.hyperparameters[component_key]
    if (
        registry == "cell_line"
        and featurizer.name == "pca"
        and featurizer.view == "methylation"
        and "n_components" in featurizer.hyperparameters
    ):
        flat["methylation_n_components"] = featurizer.hyperparameters["n_components"]
        flat.setdefault("methylation_pca_components", featurizer.hyperparameters["n_components"])
    for key, value in featurizer.hyperparameters.items():
        if key in {"featurizers", "view", "views"}:
            continue
        if "." in key or key.startswith(("featurizer.", "predictor.")):
            continue
        if (registry, featurizer.name) not in _LEGACY_FEATURIZER_FLAT_KEYS and key not in flat:
            flat.setdefault(key, value)
