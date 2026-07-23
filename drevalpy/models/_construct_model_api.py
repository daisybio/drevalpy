"""Public API for constructing DRPModel classes from modular spec strings."""

from __future__ import annotations

from drevalpy.models._native_drp_model import create_native_drp_class
from drevalpy.models.config import ModelScope
from drevalpy.models.drp_model import DRPModel


def construct_model(name: str, spec: str) -> type[DRPModel]:
    """Return a DRPModel subclass for *spec* with ``get_model_name() == name``.

    The returned class uses the ``ModelConfig`` / ``ComposedModel`` stack via
    the shared ``NativeDRPModel`` facade.
    """
    from drevalpy.models.config import ModelConfig

    config = ModelConfig.from_spec(spec)
    return create_native_drp_class(
        name,
        spec=spec,
        scope=config.scope if isinstance(config.scope, ModelScope) else ModelScope.MULTI_DRUG,
    )
