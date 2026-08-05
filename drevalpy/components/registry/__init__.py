"""Component registries for cell-line featurizers, drug featurizers, and predictors."""

from drevalpy.components.registry.base import Registry
from drevalpy.components.registry.featurizer import (
    CellLineFeaturizerRegistry,
    DrugFeaturizerRegistry,
    FeaturizerRegistry,
)
from drevalpy.components.registry.lookup import (
    clear_cell_line_featurizer_registry,
    clear_drug_featurizer_registry,
    clear_predictor_registry,
    get_cell_line_featurizer,
    get_cell_line_featurizer_metadata,
    get_drug_featurizer,
    get_drug_featurizer_metadata,
    get_predictor,
    get_predictor_metadata,
    list_cell_line_featurizer_metadata,
    list_cell_line_featurizers,
    list_drug_featurizer_metadata,
    list_drug_featurizers,
    list_predictor_metadata,
    list_predictors,
    register_cell_line_featurizer,
    register_drug_featurizer,
    register_predictor,
)
from drevalpy.components.registry.predictor import PredictorRegistry

__all__ = [
    "CellLineFeaturizerRegistry",
    "DrugFeaturizerRegistry",
    "FeaturizerRegistry",
    "PredictorRegistry",
    "Registry",
    "clear_cell_line_featurizer_registry",
    "clear_drug_featurizer_registry",
    "clear_predictor_registry",
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
