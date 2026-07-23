"""Public API for constructing DRPModel classes from modular specs."""

from __future__ import annotations

from drevalpy.models._factory_classes import _early_stopping_from_predictor, create_factory_class
from drevalpy.models._native_drp_model import create_native_drp_class
from drevalpy.models.config import ModelScope
from drevalpy.models.drp_model import DRPModel

_CONSTRUCTED_CACHE: dict[tuple[str, str], type[DRPModel]] = {}


def construct_model(name: str, spec: str | None = None) -> type[DRPModel]:
    """Return a DRPModel subclass for a zoo name or custom recipe.

    Call forms:

    - ``construct_model("ElasticNet")`` — resolve a built-in or external zoo preset
    - ``construct_model("MyModel", "scaledGeneExpression:fingerprints:elasticNet")`` —
      build a custom facade with ``get_model_name() == name``

    The returned class uses the ``ModelConfig`` / ``ComposedModel`` stack via
    the shared ``NativeDRPModel`` facade. Class-level ``early_stopping`` is
    derived from the predictor capability, matching zoo-generated classes.
    """
    from drevalpy.models.config import ModelConfig
    from drevalpy.models.zoo import list_zoo_names

    if spec is None:
        if name not in list_zoo_names(include_external=True):
            msg = (
                f"Unknown model name {name!r}. Pass a zoo preset name, or provide "
                "a recipe as the second argument: "
                'construct_model("MyModel", "cellLine:drug:predictor").'
            )
            raise ValueError(msg)
        cache_key = (name, name)
        cached = _CONSTRUCTED_CACHE.get(cache_key)
        if cached is not None:
            return cached
        cls = create_factory_class(name)
        _CONSTRUCTED_CACHE[cache_key] = cls
        return cls

    cache_key = (name, spec)
    cached = _CONSTRUCTED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    config = ModelConfig.from_spec(spec)
    cls = create_native_drp_class(
        name,
        spec=spec,
        scope=config.scope if isinstance(config.scope, ModelScope) else ModelScope.MULTI_DRUG,
        class_dict={"early_stopping": _early_stopping_from_predictor(config)},
    )
    _CONSTRUCTED_CACHE[cache_key] = cls
    return cls
