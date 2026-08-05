"""Registry singletons and public registry types.

Prefer importing from ``drevalpy.components.registry`` or the role-specific
modules (``featurizer``, ``predictor``, ``base``). This module re-exports the
singletons for existing internal imports.
"""

from drevalpy.components.registry.base import Registry
from drevalpy.components.registry.featurizer import (
    FeaturizerRegistry,
    cell_line_featurizer_registry,
    drug_featurizer_registry,
)
from drevalpy.components.registry.predictor import PredictorRegistry, predictor_registry

__all__ = [
    "FeaturizerRegistry",
    "PredictorRegistry",
    "Registry",
    "cell_line_featurizer_registry",
    "drug_featurizer_registry",
    "predictor_registry",
]
