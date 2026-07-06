"""Compatibility re-export for moved DIPK model utilities."""

from drevalpy.components.predictors.literature.impl.dipk.model_utils import (
    AttentionLayer,
    DenseLayers,
    Predictor,
)

__all__ = ["AttentionLayer", "DenseLayers", "Predictor"]
