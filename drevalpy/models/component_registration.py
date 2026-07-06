"""Legacy component registration hook (retained for import compatibility)."""

from __future__ import annotations


def register_legacy_component_predictors() -> None:
    """Literature predictors are registered directly under ``components.predictors.literature``."""


def ensure_legacy_components_registered() -> None:
    """No-op retained for backward compatibility."""
