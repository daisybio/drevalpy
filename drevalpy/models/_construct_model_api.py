"""Public API for constructing DRPModel classes from modular spec strings."""

from __future__ import annotations

from drevalpy.models._native_drp_model import create_native_drp_class
from drevalpy.models.drp_model import DRPModel


def construct_model(name: str, spec: str) -> type[DRPModel]:
    """Return a DRPModel subclass for *spec* with ``get_model_name() == name``.

    The returned class uses the existing ``ModelConfig`` / ``ComposedModel`` stack via
    ``ComponentDRPBridge`` and does not duplicate composition logic.
    """
    return create_native_drp_class(name, spec=spec)
