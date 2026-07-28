"""Hyperparameter search space utilities for internal modular composition."""

from __future__ import annotations

import copy
import re
from typing import Any

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.featurizer_label import qualified_featurizer_selector
from drevalpy.components.registry import lookup as reg
from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictorConfig

_INDEXED_FEATURIZER_KEY_RE = re.compile(
    r"^featurizer\.(?P<registry>cell_line|drug)\.(?P<name>[^.]+)\.(?P<index>\d+)\.(?P<param>.+)$"
)
_QUALIFIED_FEATURIZER_KEY_RE = re.compile(
    r"^featurizer\.(?P<registry>cell_line|drug)\.(?P<selector>[^.]+(?:\[[^\]]+\])?)\.(?P<param>.+)$"
)


def _effective_space(config_space: dict[str, dict[str, Any]] | None, cls: type[Any]) -> dict[str, Any]:
    if config_space is not None:
        return dict(config_space)
    return dict(cls.get_hyperparameter_space())


def _featurizer_prefix(registry: str, selector: str, param: str) -> str:
    return f"featurizer.{registry}.{selector}.{param}"


def _predictor_prefix(name: str, param: str) -> str:
    return f"predictor.{name}.{param}"


def _leaf_selector(featurizer: FeaturizerConfig) -> str:
    return qualified_featurizer_selector(featurizer.name, featurizer.view)


def _accepted_featurizer_selectors(featurizer: FeaturizerConfig, registry: str) -> list[str]:
    from drevalpy.components.featurizer_tree import iter_featurizer_leaves

    return [_leaf_selector(leaf) for leaf in iter_featurizer_leaves(featurizer, registry)]


def _reject_indexed_featurizer_key(key: str) -> None:
    match = _INDEXED_FEATURIZER_KEY_RE.match(key)
    if match is None:
        return
    registry = match.group("registry")
    name = match.group("name")
    param = match.group("param")
    msg = (
        f"Indexed featurizer hyperparameter keys are no longer supported: {key!r}. "
        f"Use a qualified selector such as "
        f"'featurizer.{registry}.{name}[<view>].{param}' "
        f"or 'featurizer.{registry}.{name}.{param}'."
    )
    raise ValueError(msg)


def _split_prefixed_key(key: str) -> tuple[str, str, str] | None:
    """Parse ``featurizer.<registry>.<selector>.<param>`` into parts."""
    _reject_indexed_featurizer_key(key)
    match = _QUALIFIED_FEATURIZER_KEY_RE.match(key)
    if match is None:
        return None
    return match.group("registry"), match.group("selector"), match.group("param")


def _featurizer_spaces(featurizer: FeaturizerConfig) -> dict[str, Any]:
    registry = str(featurizer.registry)
    if featurizer.name == "concatFeaturizers":
        merged: dict[str, Any] = {}
        children = featurizer.hyperparameters.get("featurizers", [])
        for child in children:
            child_cfg = FeaturizerConfig.model_validate(
                normalize_featurizer_config(child, default_registry=registry),
            )
            child_space = _featurizer_spaces(child_cfg)
            for key, spec in child_space.items():
                if key.startswith("featurizer."):
                    merged[key] = spec
                else:
                    selector = _leaf_selector(child_cfg)
                    merged[_featurizer_prefix(registry, selector, key)] = spec
        return merged

    cls = (
        reg.get_cell_line_featurizer(featurizer.name)
        if registry == "cell_line"
        else reg.get_drug_featurizer(
            featurizer.name,
        )
    )
    space = _effective_space(featurizer.hyperparameter_space, cls)
    selector = _leaf_selector(featurizer)
    return {_featurizer_prefix(registry, selector, key): value for key, value in space.items()}


def _predictor_spaces(predictor: PredictorConfig) -> dict[str, Any]:
    cls = reg.get_predictor(predictor.name)
    space = _effective_space(predictor.hyperparameter_space, cls)
    return {_predictor_prefix(predictor.name, key): value for key, value in space.items()}


def merge_model_config_spaces(config: ModelConfig) -> dict[str, Any]:
    """Merge all component spaces for a declarative model config."""
    merged: dict[str, Any] = {}
    if config.cell_line_featurizer is not None:
        merged.update(_featurizer_spaces(config.cell_line_featurizer))
    if config.drug_featurizer is not None:
        merged.update(_featurizer_spaces(config.drug_featurizer))
    merged.update(_predictor_spaces(config.predictor))
    return merged


def merge_search_spaces(
    cell_line_featurizer_space: dict[str, Any] | None = None,
    drug_featurizer_space: dict[str, Any] | None = None,
    predictor_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge component spaces into a single dict with dot-notation prefixed keys."""
    merged: dict[str, Any] = {}
    if cell_line_featurizer_space:
        for key, value in cell_line_featurizer_space.items():
            merged[f"featurizer.cell_line.{key}"] = value
    if drug_featurizer_space:
        for key, value in drug_featurizer_space.items():
            merged[f"featurizer.drug.{key}"] = value
    if predictor_space:
        for key, value in predictor_space.items():
            merged[f"predictor.{key}"] = value
    return merged


def split_hyperparameters(
    merged_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Invert merged search spaces into per-role hyperparameter dicts."""
    cell_line_hp: dict[str, Any] = {}
    drug_hp: dict[str, Any] = {}
    predictor_hp: dict[str, Any] = {}
    for key, value in merged_config.items():
        if key.startswith("featurizer.cell_line."):
            cell_line_hp[key.removeprefix("featurizer.cell_line.")] = value
        elif key.startswith("featurizer.drug."):
            drug_hp[key.removeprefix("featurizer.drug.")] = value
        elif key.startswith("predictor."):
            predictor_hp[key.removeprefix("predictor.")] = value
        else:
            predictor_hp[key] = value
    return cell_line_hp, drug_hp, predictor_hp


def _split_predictor_key(key: str) -> tuple[str, str] | None:
    parts = key.split(".")
    if len(parts) < 3 or parts[0] != "predictor":
        return None
    predictor_name, *param_parts = parts[1:]
    if not param_parts:
        return None
    return predictor_name, ".".join(param_parts)


def _merged_updates_for_featurizer(
    merged: dict[str, Any],
    *,
    registry: str,
    selector: str,
) -> dict[str, Any]:
    return {
        param: value
        for key, value in merged.items()
        if (parsed := _split_prefixed_key(key)) is not None and parsed[0] == registry and parsed[1] == selector
        for param in [parsed[2]]
    }


def _apply_hyperparameter_updates(
    featurizer: FeaturizerConfig,
    updates: dict[str, Any],
) -> FeaturizerConfig:
    if not updates:
        return featurizer
    return featurizer.model_copy(
        update={"hyperparameters": {**featurizer.hyperparameters, **updates}},
        deep=True,
    )


def _unknown_featurizer_key_message(
    key: str,
    *,
    registry: str,
    accepted: list[str],
) -> str:
    preview = ", ".join(repr(selector) for selector in accepted) if accepted else "(none)"
    return (
        f"Unknown structured featurizer hyperparameter {key!r} for registry {registry!r}. "
        f"Accepted selectors: {preview}."
    )


def _validate_featurizer_keys_for_tree(
    merged: dict[str, Any],
    featurizer: FeaturizerConfig,
) -> None:
    registry = str(featurizer.registry)
    accepted = _accepted_featurizer_selectors(featurizer, registry)
    accepted_set = set(accepted)
    for key in merged:
        if not key.startswith(f"featurizer.{registry}."):
            continue
        parsed = _split_prefixed_key(key)
        if parsed is None:
            msg = _unknown_featurizer_key_message(key, registry=registry, accepted=accepted)
            raise ValueError(msg)
        key_registry, selector, _param = parsed
        if key_registry != registry or selector not in accepted_set:
            msg = _unknown_featurizer_key_message(key, registry=registry, accepted=accepted)
            raise ValueError(msg)


def _apply_to_concat_featurizer(
    featurizer: FeaturizerConfig,
    merged: dict[str, Any],
    *,
    registry: str,
) -> FeaturizerConfig:
    children = list(featurizer.hyperparameters.get("featurizers", []))
    updated_children: list[Any] = []
    for child in children:
        child_cfg = FeaturizerConfig.model_validate(
            normalize_featurizer_config(child, default_registry=registry),
        )
        if child_cfg.name == "concatFeaturizers":
            updated = _apply_to_concat_featurizer(child_cfg, merged, registry=registry)
        else:
            child_updates = _merged_updates_for_featurizer(
                merged,
                registry=registry,
                selector=_leaf_selector(child_cfg),
            )
            updated = _apply_hyperparameter_updates(child_cfg, child_updates)
        updated_children.append(updated.model_dump())
    return featurizer.model_copy(
        update={
            "hyperparameters": {
                **featurizer.hyperparameters,
                "featurizers": updated_children,
            }
        },
        deep=True,
    )


def _apply_to_featurizer(
    featurizer: FeaturizerConfig,
    merged: dict[str, Any],
) -> FeaturizerConfig:
    registry = str(featurizer.registry)
    _validate_featurizer_keys_for_tree(merged, featurizer)
    if featurizer.name == "concatFeaturizers":
        return _apply_to_concat_featurizer(featurizer, merged, registry=registry)

    updates = _merged_updates_for_featurizer(
        merged,
        registry=registry,
        selector=_leaf_selector(featurizer),
    )
    return _apply_hyperparameter_updates(featurizer, updates)


def apply_merged_to_model_config(config: ModelConfig, merged: dict[str, Any]) -> ModelConfig:
    """Apply merged prefixed hyperparameters onto a model config."""
    result = copy.deepcopy(config)
    if result.cell_line_featurizer is not None:
        result.cell_line_featurizer = _apply_to_featurizer(result.cell_line_featurizer, merged)
    if result.drug_featurizer is not None:
        result.drug_featurizer = _apply_to_featurizer(result.drug_featurizer, merged)
    predictor_updates = {
        param: value
        for key, value in merged.items()
        if (parsed := _split_predictor_key(key)) is not None and parsed[0] == result.predictor.name
        for param in [parsed[1]]
    }
    if predictor_updates:
        result.predictor = result.predictor.model_copy(
            update={
                "hyperparameters": {
                    **result.predictor.hyperparameters,
                    **predictor_updates,
                }
            },
            deep=True,
        )
    return result


def extract_defaults(
    cell_line_featurizer_space: dict[str, Any] | None = None,
    drug_featurizer_space: dict[str, Any] | None = None,
    predictor_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pull ``default`` values from spec dicts, returning a merged flat dict."""
    defaults: dict[str, Any] = {}

    def _pull(space: dict[str, Any], prefix: str) -> None:
        for name, spec in space.items():
            if isinstance(spec, dict) and "default" in spec:
                defaults[f"{prefix}.{name}"] = spec["default"]

    if cell_line_featurizer_space:
        _pull(cell_line_featurizer_space, "featurizer.cell_line")
    if drug_featurizer_space:
        _pull(drug_featurizer_space, "featurizer.drug")
    if predictor_space:
        _pull(predictor_space, "predictor")
    return defaults


def defaults_from_merged_space(space: dict[str, Any]) -> dict[str, Any]:
    """Extract default values from a merged structured search space."""
    defaults: dict[str, Any] = {}
    for key, spec in space.items():
        if isinstance(spec, dict) and "default" in spec:
            defaults[key] = spec["default"]
    return defaults


def dict_to_ray_space(space_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert structured specs to Ray Tune distributions."""
    from ray import tune

    result: dict[str, Any] = {}
    for name, spec in space_dict.items():
        if not isinstance(spec, dict):
            result[name] = spec
            continue
        kind = spec.get("type", "categorical")
        if kind == "int":
            int_low, int_high = int(spec["low"]), int(spec["high"])
            result[name] = (
                tune.lograndint(int_low, int_high + 1)
                if spec.get("log", False)
                else tune.randint(int_low, int_high + 1)
            )
        elif kind == "float":
            float_low, float_high = float(spec["low"]), float(spec["high"])
            result[name] = (
                tune.loguniform(float_low, float_high)
                if spec.get("log", False)
                else tune.uniform(float_low, float_high)
            )
        elif kind == "categorical":
            result[name] = tune.choice(list(spec.get("choices", [])))
        else:
            msg = f"Unknown hyperparameter type {kind!r} for parameter {name!r}"
            raise ValueError(msg)
    return result
