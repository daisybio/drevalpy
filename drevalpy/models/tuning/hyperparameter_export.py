"""Export public hyperparameter mappings from model configurations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from drevalpy.components.featurizers._featurizer_tree import iter_featurizer_leaves
from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer, get_predictor
from drevalpy.models.config import FeaturizerConfig, ModelConfig

from .hyperparameter_keys import (
    HyperparameterOwnershipIndex,
    HyperparameterTarget,
    _leaf_selector,
    build_ownership_index,
)
from .search_space import (
    _featurizer_prefix,
    _predictor_prefix,
)


def _predictor_value(config: ModelConfig, param: str, predictor_cls: type[Any]) -> Any | None:
    space = (
        dict(config.predictor.hyperparameter_space)
        if config.predictor.hyperparameter_space is not None
        else dict(predictor_cls.get_hyperparameter_space())
    )
    if param in space:
        return space[param]["default"]
    defaults = predictor_cls.get_default_hyperparameters()
    if param in defaults:
        return defaults[param]
    return None


def _predictor_export_params(config: ModelConfig, predictor_cls: type[Any]) -> list[str]:
    keys = set(predictor_cls.get_default_hyperparameters())
    keys.update(predictor_cls.get_hyperparameter_space())
    if config.predictor.hyperparameter_space is not None:
        keys.update(config.predictor.hyperparameter_space)
    return sorted(keys)


def _featurizer_export_params(featurizer: FeaturizerConfig, registry: str) -> list[str]:
    cls = get_cell_line_featurizer(featurizer.name) if registry == "cell_line" else get_drug_featurizer(featurizer.name)
    space = (
        dict(featurizer.hyperparameter_space)
        if featurizer.hyperparameter_space is not None
        else dict(cls.get_hyperparameter_space())
    )
    return sorted(space)


def _featurizer_value(featurizer: FeaturizerConfig, param: str, registry: str) -> Any | None:
    cls = get_cell_line_featurizer(featurizer.name) if registry == "cell_line" else get_drug_featurizer(featurizer.name)
    space = (
        dict(featurizer.hyperparameter_space)
        if featurizer.hyperparameter_space is not None
        else dict(cls.get_hyperparameter_space())
    )
    if param not in space:
        return None
    return space[param]["default"]


def _append_export_entry(
    entries: list[tuple[HyperparameterTarget, Any]],
    *,
    qualified: str,
    index: HyperparameterOwnershipIndex,
    concrete: dict[str, Any],
    default_value: Any | None,
) -> None:
    target = index.qualified_to_target[qualified]
    if qualified in concrete:
        entries.append((target, concrete[qualified]))
        return
    if default_value is not None:
        entries.append((target, default_value))


def _collect_predictor_export_entries(
    config: ModelConfig,
    index: HyperparameterOwnershipIndex,
    concrete: dict[str, Any],
) -> list[tuple[HyperparameterTarget, Any]]:
    entries: list[tuple[HyperparameterTarget, Any]] = []
    predictor_cls = get_predictor(config.predictor.name)
    for param in _predictor_export_params(config, predictor_cls):
        qualified = _predictor_prefix(config.predictor.name, param)
        _append_export_entry(
            entries,
            qualified=qualified,
            index=index,
            concrete=concrete,
            default_value=_predictor_value(config, param, predictor_cls),
        )
    return entries


def _collect_featurizer_export_entries(
    config: ModelConfig,
    index: HyperparameterOwnershipIndex,
    concrete: dict[str, Any],
) -> list[tuple[HyperparameterTarget, Any]]:
    entries: list[tuple[HyperparameterTarget, Any]] = []
    for registry, slot_config in (
        ("cell_line", config.cell_line_featurizer),
        ("drug", config.drug_featurizer),
    ):
        if slot_config is None:
            continue
        for leaf in iter_featurizer_leaves(slot_config, registry):
            selector = _leaf_selector(leaf)
            for param in _featurizer_export_params(leaf, registry):
                qualified = _featurizer_prefix(registry, selector, param)
                _append_export_entry(
                    entries,
                    qualified=qualified,
                    index=index,
                    concrete=concrete,
                    default_value=_featurizer_value(leaf, param, registry),
                )
    return entries


def _collect_export_entries(
    config: ModelConfig,
    index: HyperparameterOwnershipIndex,
    *,
    values: dict[str, Any] | None = None,
) -> list[tuple[HyperparameterTarget, Any]]:
    concrete = values or {}
    entries = _collect_predictor_export_entries(config, index, concrete)
    entries.extend(_collect_featurizer_export_entries(config, index, concrete))
    return entries


def _compact_export_entries(
    entries: list[tuple[HyperparameterTarget, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[HyperparameterTarget, Any]]] = defaultdict(list)
    for target, value in entries:
        grouped[target.param].append((target, value))

    exported: dict[str, Any] = {}
    for param in sorted(grouped):
        owners = grouped[param]
        if len(owners) == 1:
            exported[param] = owners[0][1]
            continue
        for target, value in sorted(owners, key=lambda item: item[0].qualified_key):
            exported[target.qualified_key] = value
    return exported


def export_public_mapping(
    config: ModelConfig,
    *,
    values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export a deterministic collision-aware public hyperparameter mapping.

    :param config: Template model configuration.
    :param values: Optional concrete qualified values from a resolved config.
    :returns: Result.
    """
    index = build_ownership_index(config)
    return _compact_export_entries(_collect_export_entries(config, index, values=values))


def export_public_mapping_from_resolved(
    resolved: Any,
) -> dict[str, Any]:
    """Export public hyperparameters from a resolved instance config.

    :param resolved: ``ResolvedModelConfig`` instance.
    :returns: Compact public hyperparameter mapping.
    """
    return export_public_mapping(
        resolved.template,
        values=dict(resolved.values),
    )
