"""Composable model components: featurizers, predictors, registries, and extensions.

Model orchestration (config, zoo, and DRPModel construction) lives under
`drevalpy.models`. Experiment workflows use classes from ``construct_model``,
which compose these components.
"""

from drevalpy.components.extensions import (
    load_extension_dir,
    load_extension_file,
    load_extension_module,
    load_extensions,
)
from drevalpy.components.model_id import format_model_id, parse_model_id
from drevalpy.components.register_builtins import ensure_components_registered, register_builtin_components
from drevalpy.components.registry import (
    get_cell_line_featurizer,
    get_drug_featurizer,
    get_predictor,
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

__all__ = [
    "ensure_components_registered",
    "format_model_id",
    "get_cell_line_featurizer",
    "get_drug_featurizer",
    "get_predictor",
    "list_cell_line_featurizer_metadata",
    "list_cell_line_featurizers",
    "list_drug_featurizer_metadata",
    "list_drug_featurizers",
    "list_predictor_metadata",
    "list_predictors",
    "load_extension_dir",
    "load_extension_file",
    "load_extension_module",
    "load_extensions",
    "parse_model_id",
    "register_builtin_components",
    "register_cell_line_featurizer",
    "register_drug_featurizer",
    "register_predictor",
]
