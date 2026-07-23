"""Native structured literature predictors backed by component-owned engines."""

from drevalpy.components.predictors.literature.dipk_predictor import DIPKPredictor
from drevalpy.components.predictors.literature.molir_predictor import MOLIRPredictor
from drevalpy.components.predictors.literature.pharmaformer_predictor import PharmaFormerPredictor
from drevalpy.components.predictors.literature.precily_predictor import PrecilyPredictor
from drevalpy.components.predictors.literature.sparsego_predictor import SparseGOPredictor
from drevalpy.components.predictors.literature.srmf_predictor import SRMFPredictor
from drevalpy.components.predictors.literature.structured_engine_adapter import (
    StructuredLiteratureEnginePredictor,
    resolve_engine_cls,
)
from drevalpy.components.predictors.literature.superfeltr_predictor import SuperFELTRPredictor

# Backward-compatible alias used by older imports/tests.
_resolve_engine_cls = resolve_engine_cls

__all__ = [
    "DIPKPredictor",
    "MOLIRPredictor",
    "PharmaFormerPredictor",
    "PrecilyPredictor",
    "SRMFPredictor",
    "SparseGOPredictor",
    "StructuredLiteratureEnginePredictor",
    "SuperFELTRPredictor",
]
