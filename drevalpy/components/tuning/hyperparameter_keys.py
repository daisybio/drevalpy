"""Ownership indexes for public and structured hyperparameter keys."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from drevalpy.components.featurizer_tree import iter_featurizer_leaves
from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer, get_predictor
from drevalpy.components.tuning.compatibility_keys import LEGACY_FEATURIZER_FLAT_KEYS
from drevalpy.models.config import FeaturizerConfig, ModelConfig

from .search_space import (
    _featurizer_prefix,
    _predictor_prefix,
    _split_predictor_key,
    _split_prefixed_key,
)

_CELL_LINE_SLOT = "cell_line_featurizer"
_DRUG_SLOT = "drug_featurizer"
_PREDICTOR_SLOT = "predictor"


@dataclass(frozen=True, slots=True)
class HyperparameterTarget:
    """One public hyperparameter slot on a composed model stack."""

    slot: str
    selector: str
    param: str

    @property
    def qualified_key(self) -> str:
        """Return the fully qualified public key for this target.

        :returns: Result.
        """
        if self.slot == _PREDICTOR_SLOT:
            return _predictor_prefix(self.selector, self.param)
        return _featurizer_prefix(
            "cell_line" if self.slot == _CELL_LINE_SLOT else "drug",
            self.selector,
            self.param,
        )


@dataclass(frozen=True, slots=True)
class HyperparameterOwnershipIndex:
    """Maps short, qualified, and legacy alias keys to component targets."""

    targets: tuple[HyperparameterTarget, ...]
    qualified_to_target: dict[str, HyperparameterTarget]
    short_to_targets: dict[str, tuple[HyperparameterTarget, ...]]
    alias_to_qualified: dict[str, str]


def _leaf_selector(featurizer: FeaturizerConfig) -> str:
    from drevalpy.components.featurizer_label import qualified_featurizer_selector

    return qualified_featurizer_selector(featurizer.name, featurizer.view)


def _predictor_accepted_keys(predictor_cls: type[Any]) -> set[str]:
    keys = set(predictor_cls.get_default_hyperparameters())
    keys.update(predictor_cls.get_hyperparameter_space())
    non_tunable = getattr(predictor_cls, "non_tunable_hyperparameters", None)
    if isinstance(non_tunable, dict):
        keys.update(non_tunable)
    elif isinstance(non_tunable, (set, frozenset, list, tuple)):
        keys.update(str(key) for key in non_tunable)
    return keys


def _featurizer_accepted_keys(featurizer: FeaturizerConfig, registry: str) -> set[str]:
    cls = get_cell_line_featurizer(featurizer.name) if registry == "cell_line" else get_drug_featurizer(featurizer.name)
    return set(cls.get_hyperparameter_space())


def _append_featurizer_targets(
    targets: list[HyperparameterTarget],
    featurizer: FeaturizerConfig,
    *,
    slot: str,
    registry: str,
) -> None:
    selector = _leaf_selector(featurizer)
    for param in sorted(_featurizer_accepted_keys(featurizer, registry)):
        targets.append(HyperparameterTarget(slot=slot, selector=selector, param=param))


def _legacy_alias_targets(targets: tuple[HyperparameterTarget, ...]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    qualified_set = {target.qualified_key for target in targets}

    methylation_key = _featurizer_prefix("cell_line", "pca[methylation]", "n_components")
    if methylation_key in qualified_set:
        aliases["methylation_n_components"] = methylation_key
        aliases["methylation_pca_components"] = methylation_key

    for target in targets:
        if target.slot not in {_CELL_LINE_SLOT, _DRUG_SLOT}:
            continue
        registry = "cell_line" if target.slot == _CELL_LINE_SLOT else "drug"
        legacy = LEGACY_FEATURIZER_FLAT_KEYS.get((registry, target.selector), {})
        for component_key, flat_key in legacy.items():
            if target.param == component_key:
                aliases[flat_key] = target.qualified_key
    return aliases


def build_ownership_index(config: ModelConfig) -> HyperparameterOwnershipIndex:
    """Build ownership indexes for every accepted public hyperparameter.

    :param config: config.
    :returns: Result.
    """
    targets: list[HyperparameterTarget] = []

    predictor_cls = get_predictor(config.predictor.name)
    for param in sorted(_predictor_accepted_keys(predictor_cls)):
        targets.append(
            HyperparameterTarget(
                slot=_PREDICTOR_SLOT,
                selector=config.predictor.name,
                param=param,
            ),
        )

    if config.cell_line_featurizer is not None:
        for leaf in iter_featurizer_leaves(config.cell_line_featurizer, "cell_line"):
            _append_featurizer_targets(
                targets,
                leaf,
                slot=_CELL_LINE_SLOT,
                registry="cell_line",
            )

    if config.drug_featurizer is not None:
        for leaf in iter_featurizer_leaves(config.drug_featurizer, "drug"):
            _append_featurizer_targets(
                targets,
                leaf,
                slot=_DRUG_SLOT,
                registry="drug",
            )

    qualified_to_target = {target.qualified_key: target for target in targets}
    short_groups: dict[str, list[HyperparameterTarget]] = defaultdict(list)
    for target in targets:
        short_groups[target.param].append(target)

    target_tuple = tuple(targets)
    return HyperparameterOwnershipIndex(
        targets=target_tuple,
        qualified_to_target=qualified_to_target,
        short_to_targets={key: tuple(group) for key, group in short_groups.items()},
        alias_to_qualified=_legacy_alias_targets(target_tuple),
    )


def _parse_qualified_key(key: str, config: ModelConfig) -> str | None:
    predictor_parsed = _split_predictor_key(key)
    if predictor_parsed is not None:
        predictor_name, _param = predictor_parsed
        if predictor_name != config.predictor.name:
            msg = (
                f"Unknown hyperparameter {key!r}: predictor {predictor_name!r} "
                f"does not match stack predictor {config.predictor.name!r}."
            )
            raise ValueError(msg)
        return key

    featurizer_parsed = _split_prefixed_key(key)
    if featurizer_parsed is not None:
        registry, selector, param = featurizer_parsed
        return _featurizer_prefix(registry, selector, param)

    if key.startswith(f"{_PREDICTOR_SLOT}."):
        msg = f"Unknown hyperparameter {key!r}: expected predictor.<name>.<param>."
        raise ValueError(msg)
    if key.startswith((_CELL_LINE_SLOT + ".", _DRUG_SLOT + ".")):
        msg = f"Unknown hyperparameter {key!r}: expected <slot>.<selector>.<param>."
        raise ValueError(msg)
    return None


def _resolve_one_public_key(
    key: str,
    config: ModelConfig,
    index: HyperparameterOwnershipIndex,
) -> str:
    if key in index.alias_to_qualified:
        return index.alias_to_qualified[key]
    if key in index.qualified_to_target:
        return key
    if "." in key:
        parsed = _parse_qualified_key(key, config)
        if parsed is None or parsed not in index.qualified_to_target:
            msg = f"Unknown hyperparameter {key!r} for this model stack."
            raise ValueError(msg)
        return parsed
    if key in index.short_to_targets:
        owners = index.short_to_targets[key]
        if len(owners) > 1:
            alternatives = ", ".join(repr(owner.qualified_key) for owner in owners)
            msg = f"Ambiguous hyperparameter {key!r}. Use one of: {alternatives}."
            raise ValueError(msg)
        return owners[0].qualified_key
    msg = f"Unknown hyperparameter {key!r} for this model stack."
    raise ValueError(msg)


def resolve_to_qualified_mapping(
    config: ModelConfig,
    mapping: dict[str, Any],
    index: HyperparameterOwnershipIndex,
    *,
    reserved_keys: frozenset[str],
) -> dict[str, Any]:
    """Resolve a public mapping to qualified keys with strict collision checks.

    :param config: config.
    :param mapping: mapping.
    :param index: index.
    :param reserved_keys: reserved keys.
    :returns: Result.
    :raises ValueError: Raised on invalid input.
    """
    qualified: dict[str, Any] = {}
    seen_targets: dict[HyperparameterTarget, str] = {}

    for key, value in mapping.items():
        if key in reserved_keys:
            continue
        qualified_key = _resolve_one_public_key(key, config, index)
        target = index.qualified_to_target[qualified_key]
        if target in seen_targets:
            previous = seen_targets[target]
            msg = f"Duplicate hyperparameter assignment for {qualified_key!r} from {previous!r} and {key!r}."
            raise ValueError(msg)
        seen_targets[target] = key
        qualified[qualified_key] = value

    return qualified


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
    include_view_keys: bool = False,
    values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export a deterministic collision-aware public hyperparameter mapping.

    :param config: Template model configuration.
    :param include_view_keys: include view keys.
    :param values: Optional concrete qualified values from a resolved config.
    :returns: Result.
    """
    index = build_ownership_index(config)
    exported = _compact_export_entries(_collect_export_entries(config, index, values=values))
    if include_view_keys:
        cell_line_views = config.cell_line_views()
        drug_views = config.drug_views()
        if cell_line_views:
            exported["cell_line_views"] = cell_line_views
        if drug_views:
            exported["drug_views"] = drug_views
    return exported


def export_public_mapping_from_resolved(
    resolved: Any,
    *,
    include_view_keys: bool = False,
) -> dict[str, Any]:
    """Export public hyperparameters from a resolved instance config.

    :param resolved: ``ResolvedModelConfig`` instance.
    :param include_view_keys: Whether to include view keys.
    :returns: Compact public hyperparameter mapping.
    """
    exported = export_public_mapping(
        resolved.template,
        values=dict(resolved.values),
    )
    if include_view_keys:
        cell_line_views = resolved.cell_line_views()
        drug_views = resolved.drug_views()
        if cell_line_views:
            exported["cell_line_views"] = cell_line_views
        if drug_views:
            exported["drug_views"] = drug_views
    return exported


def validate_merged_mapping(config: ModelConfig, merged: dict[str, Any]) -> None:
    """Reject unknown or malformed qualified hyperparameter keys.

    :param config: config.
    :param merged: merged.
    :raises ValueError: Raised on invalid input.
    """
    from .search_space import _reject_indexed_featurizer_key

    index = build_ownership_index(config)
    for key in merged:
        _reject_indexed_featurizer_key(key)
        if key not in index.qualified_to_target:
            msg = f"Unknown hyperparameter {key!r} for this model stack."
            raise ValueError(msg)
