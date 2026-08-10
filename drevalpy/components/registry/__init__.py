"""Component registries for cell-line featurizers, drug featurizers, and predictors."""

from .base import Registry
from .featurizer_registration import (
    get_cell_line_featurizer,
    get_cell_line_featurizer_metadata,
    get_drug_featurizer,
    get_drug_featurizer_metadata,
    list_cell_line_featurizer_metadata,
    list_cell_line_featurizers,
    list_drug_featurizer_metadata,
    list_drug_featurizers,
    register_cell_line_featurizer,
    register_drug_featurizer,
)
from .featurizer_registry import FeaturizerRegistry
from .predictor_registration import (
    get_predictor,
    get_predictor_metadata,
    list_predictor_metadata,
    list_predictors,
    register_predictor,
)
from .predictor_registry import PredictorRegistry
from .register_builtins import register_builtin_components

register_builtin_components()

__all__ = [
    "FeaturizerRegistry",
    "PredictorRegistry",
    "Registry",
    "get_cell_line_featurizer",
    "get_cell_line_featurizer_metadata",
    "get_drug_featurizer",
    "get_drug_featurizer_metadata",
    "get_predictor",
    "get_predictor_metadata",
    "list_cell_line_featurizer_metadata",
    "list_cell_line_featurizers",
    "list_drug_featurizer_metadata",
    "list_drug_featurizers",
    "list_predictor_metadata",
    "list_predictors",
    "register_cell_line_featurizer",
    "register_drug_featurizer",
    "register_predictor",
]
