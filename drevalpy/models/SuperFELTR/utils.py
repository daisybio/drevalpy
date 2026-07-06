"""Compatibility re-export for moved SuperFELTR utilities."""

from drevalpy.components.predictors.literature.impl.superfeltr.utils import (
    SuperFELTEncoder,
    SuperFELTRegressor,
    train_superfeltr_model,
)

__all__ = ["SuperFELTEncoder", "SuperFELTRegressor", "train_superfeltr_model"]
