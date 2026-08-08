"""Public API for constructing DRPModel classes from modular specs."""

from __future__ import annotations

import functools
import json
from typing import Any

from drevalpy.models.config import ModelConfig, ModelScope, from_spec
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.models.drp_model import DRPModel


def _as_template(config: ModelConfig | ResolvedModelConfig) -> ModelConfig:
    if isinstance(config, ResolvedModelConfig):
        return config.template
    return config


def _resolve_base_config(name: str, spec: str | ModelConfig | ResolvedModelConfig | None) -> ModelConfig:
    if spec is None:
        from drevalpy.models.zoo import list_zoo_names

        if name not in list_zoo_names(include_external=True):
            msg = (
                f"Unknown model name {name!r}. Pass a zoo preset name, or provide "
                "a recipe/config as the second argument: "
                'construct_model("MyModel", "cellLine:drug:predictor").'
            )
            raise ValueError(msg)
        config = from_spec(name)
    elif isinstance(spec, ResolvedModelConfig):
        config = ModelConfig.model_validate(spec.template.model_dump(mode="python"))
    elif isinstance(spec, ModelConfig):
        config = ModelConfig.model_validate(spec.model_dump(mode="python"))
    else:
        config = from_spec(spec)
    return _as_template(config)


@functools.cache
def _generate_model_class(name: str, config_json: str) -> type[DRPModel]:
    config = ModelConfig.model_validate_json(config_json)
    attrs: dict[str, Any] = {
        "_model_name": name,
        "_base_model_config": config,
    }
    cls = type(name, (DRPModel,), attrs)
    cls.__module__ = "drevalpy.models"
    return cls


def construct_model(name: str, spec: str | ModelConfig | ResolvedModelConfig | None = None) -> type[DRPModel]:
    """Return a ``DRPModel`` subclass for a zoo name, recipe, or ``ModelConfig``.

    Call forms:

    - ``construct_model("ElasticNet")`` — resolve a built-in or external zoo preset
    - ``construct_model("MyModel", "scaledGeneExpression:fingerprints:elasticNet")`` —
      build a custom class with ``get_model_name() == "MyModel"``
    - ``construct_model("MyModel", config)`` — same with an already-built ``ModelConfig``

    The returned class is a thin metadata-only subclass of the concrete ``DRPModel``.
    Instantiating it with optional flat hyperparameters creates a fresh runtime instance.

    :param name: Model identity for the generated class, or a built-in zoo preset name when ``spec`` is omitted.
    :param spec: Optional recipe string, ``ModelConfig``, or ``None`` to resolve ``name`` from the zoo.
    :returns: Generated ``DRPModel`` subclass bound to the resolved config.
    """
    config = _resolve_base_config(name, spec)
    config_json = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return _generate_model_class(name, config_json)


def build_builtin_factory_tables() -> tuple[
    dict[str, type[DRPModel]],
    dict[str, type[DRPModel]],
    dict[str, type[DRPModel]],
]:
    """Build multi/single/all factory mappings for built-in zoo names only.

    :returns: Tuple of multi-drug, single-drug, and combined factory mappings.
    """
    from drevalpy.models.zoo import get_zoo_config, list_zoo_names

    multi: dict[str, type[DRPModel]] = {}
    single: dict[str, type[DRPModel]] = {}
    for factory_name in list_zoo_names(include_external=False):
        config = get_zoo_config(factory_name)
        cls = construct_model(factory_name)
        if config.scope == ModelScope.SINGLE_DRUG:
            single[factory_name] = cls
        else:
            multi[factory_name] = cls
    return multi, single, {**multi, **single}
