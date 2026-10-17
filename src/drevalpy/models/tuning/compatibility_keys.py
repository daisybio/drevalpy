"""Config-to-public featurizer translation helpers."""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizers._featurizer_tree import iter_featurizer_leaves
from drevalpy.models.config import FeaturizerConfig
from drevalpy.registry.cell_line_featurizer import get as get_cell_line_featurizer
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer


def _featurizer_space_keys(featurizer: FeaturizerConfig, registry: str) -> set[str]:
    cls = get_cell_line_featurizer(featurizer.name) if registry == "cell_line" else get_drug_featurizer(featurizer.name)
    space = (
        dict(featurizer.hyperparameter_space)
        if featurizer.hyperparameter_space is not None
        else dict(cls.get_hyperparameter_space())
    )
    return set(space)


def _is_exportable_space_flat_key(key: str, flat: dict[str, Any]) -> bool:
    if key in {"featurizers", "view", "views"}:
        return False
    if "." in key or key.startswith(("cell_line_featurizer.", "drug_featurizer.", "predictor.")):
        return False
    return key not in flat


def _append_space_default_flat_keys(
    flat: dict[str, Any],
    featurizer: FeaturizerConfig,
    registry: str,
) -> None:
    cls = get_cell_line_featurizer(featurizer.name) if registry == "cell_line" else get_drug_featurizer(featurizer.name)
    space = (
        dict(featurizer.hyperparameter_space)
        if featurizer.hyperparameter_space is not None
        else dict(cls.get_hyperparameter_space())
    )
    for key, spec in space.items():
        if not _is_exportable_space_flat_key(key, flat):
            continue
        flat.setdefault(key, spec["default"])
        if featurizer.name == "pca" and featurizer.view == "methylation" and key == "n_components":
            flat["methylation_n_components"] = spec["default"]
            flat.setdefault("methylation_pca_components", spec["default"])


def append_featurizer_flat_keys(
    flat: dict[str, Any],
    featurizer: FeaturizerConfig | None,
    registry: str,
) -> None:
    """Append tunable featurizer defaults into a public flat dict.

    Architecture-only featurizer kwargs stay on the config tree and are not flattened.
    Concrete selected values live on ``ResolvedModelConfig`` and are exported elsewhere.

    :param flat: Mutable public flat hyperparameter mapping to extend in place.
    :param featurizer: Featurizer config subtree to flatten, or ``None``.
    :param registry: Registry slot name (``cell_line`` or ``drug``).
    """
    if featurizer is None:
        return
    for leaf in iter_featurizer_leaves(featurizer, registry):
        _append_space_default_flat_keys(flat, leaf, registry)
        _ = _featurizer_space_keys(leaf, registry)
