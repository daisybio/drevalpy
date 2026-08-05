"""Public API for constructing DRPModel classes from modular specs."""

from __future__ import annotations

import json
from typing import Any

from drevalpy.models.config import ModelConfig, ModelScope
from drevalpy.models.drp_model import DRPModel

_CONSTRUCTED_CACHE: dict[tuple[str, str], type[DRPModel]] = {}


def _canonical_config_key(config: ModelConfig) -> str:
    return json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _resolve_base_config(name: str, spec: str | ModelConfig | None) -> ModelConfig:
    if spec is None:
        from drevalpy.models.zoo import list_zoo_names

        if name not in list_zoo_names(include_external=True):
            msg = (
                f"Unknown model name {name!r}. Pass a zoo preset name, or provide "
                "a recipe/config as the second argument: "
                'construct_model("MyModel", "cellLine:drug:predictor").'
            )
            raise ValueError(msg)
        config = ModelConfig.from_spec(name)
    elif isinstance(spec, ModelConfig):
        config = spec.model_copy(deep=True)
    else:
        config = ModelConfig.from_spec(spec)
    config.validate()
    return config


def _generate_model_class(name: str, config: ModelConfig) -> type[DRPModel]:
    attrs: dict[str, Any] = {
        "_model_name": name,
        "_base_model_config": config.model_copy(deep=True),
    }
    cls = type(name, (DRPModel,), attrs)
    cls.__module__ = "drevalpy.models"
    return cls


def construct_model(name: str, spec: str | ModelConfig | None = None) -> type[DRPModel]:
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
    cache_key = (name, _canonical_config_key(config))
    cached = _CONSTRUCTED_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cls = _generate_model_class(name, config)
    _CONSTRUCTED_CACHE[cache_key] = cls
    return cls


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
