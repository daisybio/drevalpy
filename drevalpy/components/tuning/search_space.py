"""Hyperparameter search space utilities for internal modular composition."""

from __future__ import annotations

import copy
from typing import Any

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.registry import lookup as reg
from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictorConfig


def _effective_space(config_space: dict[str, dict[str, Any]] | None, cls: type[Any]) -> dict[str, Any]:
    if config_space is not None:
        return dict(config_space)
    return dict(cls.get_hyperparameter_space())


def _featurizer_prefix(registry: str, name: str, index: int, param: str) -> str:
    return f"featurizer.{registry}.{name}.{index}.{param}"


def _predictor_prefix(name: str, param: str) -> str:
    return f"predictor.{name}.{param}"


def _featurizer_spaces(featurizer: FeaturizerConfig) -> dict[str, Any]:
    registry = str(featurizer.registry)
    if featurizer.name == "concatFeaturizers":
        merged: dict[str, Any] = {}
        children = featurizer.hyperparameters.get("featurizers", [])
        name_counts: dict[str, int] = {}
        for child in children:
            child_cfg = FeaturizerConfig.model_validate(
                normalize_featurizer_config(child, default_registry=registry),
            )
            child_name = child_cfg.name
            index = name_counts.get(child_name, 0)
            name_counts[child_name] = index + 1
            child_space = _featurizer_spaces(child_cfg)
            for key, spec in child_space.items():
                if key.startswith("featurizer."):
                    merged[key] = spec
                else:
                    merged[_featurizer_prefix(registry, child_name, index, key)] = spec
        return merged

    cls = (
        reg.get_cell_line_featurizer(featurizer.name)
        if registry == "cell_line"
        else reg.get_drug_featurizer(
            featurizer.name,
        )
    )
    space = _effective_space(featurizer.hyperparameter_space, cls)
    return {_featurizer_prefix(registry, featurizer.name, 0, key): value for key, value in space.items()}


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


def _split_prefixed_key(key: str) -> tuple[str, str, str, int, str] | None:
    parts = key.split(".")
    if len(parts) < 5 or parts[0] != "featurizer":
        return None
    registry, name, index_str, *param_parts = parts[1:]
    if not param_parts:
        return None
    try:
        index = int(index_str)
    except ValueError:
        return None
    return registry, name, index_str, index, ".".join(param_parts)


def _split_predictor_key(key: str) -> tuple[str, str] | None:
    parts = key.split(".")
    if len(parts) < 3 or parts[0] != "predictor":
        return None
    predictor_name, *param_parts = parts[1:]
    if not param_parts:
        return None
    return predictor_name, ".".join(param_parts)


def _apply_to_featurizer(
    featurizer: FeaturizerConfig,
    merged: dict[str, Any],
) -> FeaturizerConfig:
    registry = str(featurizer.registry)
    if featurizer.name == "concatFeaturizers":
        children = list(featurizer.hyperparameters.get("featurizers", []))
        updated_children: list[Any] = []
        name_counts: dict[str, int] = {}
        for child in children:
            child_cfg = FeaturizerConfig.model_validate(
                normalize_featurizer_config(child, default_registry=registry),
            )
            child_name = child_cfg.name
            index = name_counts.get(child_name, 0)
            name_counts[child_name] = index + 1
            child_updates = {
                param: value
                for key, value in merged.items()
                if (parsed := _split_prefixed_key(key)) is not None
                and parsed[0] == registry
                and parsed[1] == child_name
                and parsed[3] == index
                for param in [parsed[4]]
            }
            if child_updates:
                child_cfg = child_cfg.model_copy(
                    update={"hyperparameters": {**child_cfg.hyperparameters, **child_updates}},
                    deep=True,
                )
            updated_children.append(child_cfg.model_dump())
        return featurizer.model_copy(
            update={"hyperparameters": {**featurizer.hyperparameters, "featurizers": updated_children}},
            deep=True,
        )

    updates = {
        param: value
        for key, value in merged.items()
        if (parsed := _split_prefixed_key(key)) is not None
        and parsed[0] == registry
        and parsed[1] == featurizer.name
        and parsed[3] == 0
        for param in [parsed[4]]
    }
    if updates:
        return featurizer.model_copy(
            update={"hyperparameters": {**featurizer.hyperparameters, **updates}},
            deep=True,
        )
    return featurizer


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
            update={"hyperparameters": {**result.predictor.hyperparameters, **predictor_updates}},
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
