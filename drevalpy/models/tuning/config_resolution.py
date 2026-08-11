"""Resolve structured defaults and search spaces for DRPModel classes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from drevalpy.components.featurizers._featurizer_tree import iter_featurizer_leaves
from drevalpy.models.config import FeaturizerConfig, ModelConfig
from drevalpy.models.config.resolved import ResolvedModelConfig

from .search_space import (
    resolve_model_config,
)


def iter_featurizer_configs(config: ModelConfig) -> Iterator[FeaturizerConfig]:
    """Yield every leaf featurizer config in a model config."""
    for featurizer in (config.cell_line_featurizer, config.drug_featurizer):
        if featurizer is None:
            continue
        yield from iter_featurizer_leaves(featurizer, str(featurizer.registry))


def default_config_for_drp_model(model_class: type[Any]) -> ResolvedModelConfig | None:
    """Return a resolved config with structured defaults applied.

    :param model_class: Public ``DRPModel`` subclass.

    :returns: Resolved config with component defaults, or ``None`` when the model has no modular config.
    """
    config = model_class._resolve_base_config()
    if config is None:
        return None
    return resolve_model_config(config)


def tuned_config_for_drp_model(
    model_class: type[Any],
    merged_sample: dict[str, Any],
) -> ResolvedModelConfig | None:
    """Apply a structured Ray/Optuna sample onto the base model template.

    :param model_class: Public ``DRPModel`` subclass.
    :param merged_sample: Flat structured hyperparameter sample from Ray Tune.

    :returns: Resolved ``ResolvedModelConfig``, or ``None`` when the model has no modular config.
    """
    config = model_class._resolve_base_config()
    if config is None:
        return None
    return resolve_model_config(config, merged_sample)


def construct_drp_model_from_config(model_class: type[Any], config: ModelConfig | ResolvedModelConfig) -> Any:
    """Construct a public DRPModel instance from a template or resolved config.

    :param model_class: Public ``DRPModel`` subclass.
    :param config: Template or resolved modular configuration.

    :returns: Instantiated model object.
    """
    from_resolved = getattr(model_class, "_from_resolved_config", None)
    if callable(from_resolved):
        if isinstance(config, ModelConfig):
            return from_resolved(resolve_model_config(config))
        return from_resolved(config)
    from .public_flat import public_hyperparameters_from_config

    return model_class(public_hyperparameters_from_config(config))


def structured_space_for_drp_model(model_class: type[Any]) -> dict[str, Any]:
    """Return the merged structured search space for a DRPModel class.

    :param model_class: Public ``DRPModel`` subclass.

    :returns: Flat search-space dict with prefixed component keys.
    """
    return model_class.get_structured_hyperparameter_space()


def default_hyperparameters_for_drp_model(model_class: type[Any]) -> dict[str, Any]:
    """Return default hyperparameters used by ``model_class()``.

    :param model_class: Public ``DRPModel`` subclass.

    :returns: Public flat hyperparameter mapping for the model's default config.
    """
    return model_class.get_default_hyperparameters()


def has_tunable_hyperparameters(model_class: type[Any]) -> bool:
    """Return whether the model exposes a non-empty structured search space.

    :param model_class: Public ``DRPModel`` subclass.

    :returns: ``True`` when at least one tunable parameter is declared.
    """
    return bool(model_class.get_structured_hyperparameter_space())


def assert_component_local_hyperparameters(config: ModelConfig | ResolvedModelConfig) -> None:
    """Raise if namespaced keys leaked into component-local hyperparameter dicts.

    For templates this is a no-op (templates store no concrete values). For
    resolved configs, qualified keys in ``values`` are expected.

    :param config: Model configuration to validate.

    :raises AssertionError: If a featurizer or predictor hyperparameter dict contains
    """
    if isinstance(config, ResolvedModelConfig):
        for key in config.values:
            if key.count(".") < 2:
                msg = f"resolved hyperparameter key {key!r} must be qualified"
                raise AssertionError(msg)
        return
    # Templates have no concrete hyperparameters by design.
    for _featurizer in iter_featurizer_configs(config):
        pass
