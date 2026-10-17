"""Hyperparameter search space utilities for internal modular composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from drevalpy.components.contracts.hyperparameter_space import validate_hyperparameter_space
from drevalpy.components.featurizers._featurizer_label import qualified_featurizer_selector
from drevalpy.components.featurizers._featurizer_tree import iter_featurizer_leaves
from drevalpy.models._hp_key_grammar import (
    CELL_LINE_SLOT,
    DRUG_SLOT,
    PREDICTOR_SLOT,
    featurizer_prefix,
    is_featurizer_slot_key,
    predictor_prefix,
)
from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictorConfig
from drevalpy.models.config._hp_key_validation import validate_merged_mapping
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.registry.cell_line_featurizer import get as get_cell_line_featurizer
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.registry.predictor import get as get_predictor


def _effective_space(config_space: Mapping[str, Any] | None, cls: type[Any]) -> dict[str, Any]:
    if config_space is not None:
        return {key: dict(value) if isinstance(value, Mapping) else value for key, value in config_space.items()}
    return dict(cls.get_hyperparameter_space())


def _leaf_selector(featurizer: FeaturizerConfig) -> str:
    return qualified_featurizer_selector(featurizer.name, featurizer.view)


def _accepted_featurizer_selectors(featurizer: FeaturizerConfig, registry: str) -> list[str]:
    return [_leaf_selector(leaf) for leaf in iter_featurizer_leaves(featurizer, registry)]


def _merge_concat_child_spaces(
    featurizer: FeaturizerConfig,
    registry: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for child_cfg in featurizer.featurizers or ():
        child_space = _featurizer_spaces(child_cfg)
        for key, spec in child_space.items():
            if is_featurizer_slot_key(key):
                merged[key] = spec
            else:
                selector = _leaf_selector(child_cfg)
                merged[featurizer_prefix(registry, selector, key)] = spec
    return merged


def _leaf_featurizer_spaces(featurizer: FeaturizerConfig, registry: str) -> dict[str, Any]:
    cls = (
        get_cell_line_featurizer(featurizer.name)
        if registry == "cell_line"
        else get_drug_featurizer(
            featurizer.name,
        )
    )
    space = _effective_space(
        dict(featurizer.hyperparameter_space) if featurizer.hyperparameter_space is not None else None,
        cls,
    )
    selector = _leaf_selector(featurizer)
    return {featurizer_prefix(registry, selector, key): value for key, value in space.items()}


def _featurizer_spaces(featurizer: FeaturizerConfig) -> dict[str, Any]:
    registry = str(featurizer.registry)
    if featurizer.name == "concatFeaturizers":
        return _merge_concat_child_spaces(featurizer, registry)
    return _leaf_featurizer_spaces(featurizer, registry)


def _predictor_spaces(predictor: PredictorConfig) -> dict[str, Any]:
    cls = get_predictor(predictor.name)
    space = _effective_space(
        dict(predictor.hyperparameter_space) if predictor.hyperparameter_space is not None else None,
        cls,
    )
    return {predictor_prefix(predictor.name, key): value for key, value in space.items()}


def merge_model_config_spaces(config: ModelConfig) -> dict[str, Any]:
    """Merge all component spaces for a declarative model config.

    :param config: config.
    :returns: Result.
    """
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
    """Merge component spaces into a single dict with dot-notation prefixed keys.

    :param cell_line_featurizer_space: cell line featurizer space.
    :param drug_featurizer_space: drug featurizer space.
    :param predictor_space: predictor space.
    :returns: Result.
    """
    merged: dict[str, Any] = {}
    if cell_line_featurizer_space:
        for key, value in cell_line_featurizer_space.items():
            merged[f"{CELL_LINE_SLOT}.{key}"] = value
    if drug_featurizer_space:
        for key, value in drug_featurizer_space.items():
            merged[f"{DRUG_SLOT}.{key}"] = value
    if predictor_space:
        for key, value in predictor_space.items():
            merged[f"{PREDICTOR_SLOT}.{key}"] = value
    return merged


def split_hyperparameters(
    merged_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Invert merged search spaces into per-role hyperparameter dicts.

    :param merged_config: merged config.
    :returns: Result.
    """
    cell_line_hp: dict[str, Any] = {}
    drug_hp: dict[str, Any] = {}
    predictor_hp: dict[str, Any] = {}
    for key, value in merged_config.items():
        if key.startswith(f"{CELL_LINE_SLOT}."):
            cell_line_hp[key.removeprefix(f"{CELL_LINE_SLOT}.")] = value
        elif key.startswith(f"{DRUG_SLOT}."):
            drug_hp[key.removeprefix(f"{DRUG_SLOT}.")] = value
        elif key.startswith(f"{PREDICTOR_SLOT}."):
            predictor_hp[key.removeprefix(f"{PREDICTOR_SLOT}.")] = value
        else:
            predictor_hp[key] = value
    return cell_line_hp, drug_hp, predictor_hp


def resolve_model_config(
    template: ModelConfig,
    overrides: dict[str, Any] | None = None,
    *,
    include_defaults: bool = True,
) -> ResolvedModelConfig:
    """Build a resolved instance config from a template and qualified overrides.

    :param template: Immutable class-level ``ModelConfig``.
    :param overrides: Qualified concrete values to apply on top of defaults.
    :param include_defaults: When ``True``, fill omitted keys from effective spaces.
    :returns: Validated ``ResolvedModelConfig``.
    """
    qualified = dict(overrides or {})
    if include_defaults:
        defaults = defaults_from_merged_space(merge_model_config_spaces(template))
        values = {**defaults, **qualified}
    else:
        values = qualified
    validate_merged_mapping(template, values)
    return ResolvedModelConfig(template=template, values=values)


def apply_merged_to_model_config(config: ModelConfig, merged: dict[str, Any]) -> ResolvedModelConfig:
    """Apply merged prefixed hyperparameters onto a model template.

    Historically returned a ``ModelConfig`` with concrete values written into
    component ``hyperparameters``. It now returns a ``ResolvedModelConfig``
    and leaves the template unchanged.

    :param config: Immutable model template.
    :param merged: Qualified concrete hyperparameter mapping.
    :returns: Resolved instance configuration.
    """
    return resolve_model_config(config, merged, include_defaults=True)


def extract_defaults(
    cell_line_featurizer_space: dict[str, Any] | None = None,
    drug_featurizer_space: dict[str, Any] | None = None,
    predictor_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pull ``default`` values from spec dicts, returning a merged flat dict.

    :param cell_line_featurizer_space: cell line featurizer space.
    :param drug_featurizer_space: drug featurizer space.
    :param predictor_space: predictor space.
    :returns: Result.
    """
    defaults: dict[str, Any] = {}

    def _pull(space: dict[str, Any], prefix: str) -> None:
        validate_hyperparameter_space(space, context=f"hyperparameter space under {prefix!r}")
        for name, spec in space.items():
            defaults[f"{prefix}.{name}"] = spec["default"]

    if cell_line_featurizer_space:
        _pull(cell_line_featurizer_space, CELL_LINE_SLOT)
    if drug_featurizer_space:
        _pull(drug_featurizer_space, DRUG_SLOT)
    if predictor_space:
        _pull(predictor_space, PREDICTOR_SLOT)
    return defaults


def defaults_from_merged_space(space: dict[str, Any]) -> dict[str, Any]:
    """Extract default values from a merged structured search space.

    :param space: space.
    :returns: Result.
    """
    validate_hyperparameter_space(space, context="merged hyperparameter space")
    return {key: spec["default"] for key, spec in space.items()}


def sample_from_optuna_trial(trial: Any, space_dict: dict[str, Any]) -> dict[str, Any]:
    """Sample hyperparameters from an Optuna trial using the structured search space.

    :param trial: An ``optuna.Trial`` instance.
    :param space_dict: Structured hyperparameter space with entries like
        ``{"alpha": {"type": "float", "low": 0.001, "high": 10.0, "log": True}}``.
    :returns: Flat dict of sampled concrete hyperparameter values.
    :raises ValueError: If a parameter spec has an unknown type.
    """
    result: dict[str, Any] = {}
    for name, spec in space_dict.items():
        if not isinstance(spec, Mapping):
            result[name] = spec
            continue
        kind = spec.get("type", "categorical")
        if kind == "int":
            result[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=spec.get("log", False))
        elif kind == "float":
            result[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=spec.get("log", False)
            )
        elif kind == "categorical":
            result[name] = trial.suggest_categorical(name, list(spec.get("choices", [])))
        elif kind == "pow2":
            exp = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
            result[name] = 2**exp
        else:
            msg = f"Unknown hyperparameter type {kind!r} for parameter {name!r}"
            raise ValueError(msg)
    return result
