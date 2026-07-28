"""Compatibility re-exports for literature engine adapters."""

from drevalpy.components.predictors.literature._engine_resolve import (
    DISCOVERED_HYPERPARAMETERS_KEY,
    ENGINE_MODULES,
    resolve_engine_cls,
)
from drevalpy.components.predictors.literature.block_engine_adapter import BlockLiteratureEnginePredictor
from drevalpy.components.predictors.literature.raw_engine_adapter import RawLiteratureEnginePredictor

# Historical alias used by older imports/tests.
StructuredLiteratureEnginePredictor = BlockLiteratureEnginePredictor

__all__ = [
    "DISCOVERED_HYPERPARAMETERS_KEY",
    "ENGINE_MODULES",
    "BlockLiteratureEnginePredictor",
    "RawLiteratureEnginePredictor",
    "StructuredLiteratureEnginePredictor",
    "resolve_engine_cls",
]
