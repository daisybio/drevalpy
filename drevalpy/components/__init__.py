"""Composable model components: featurizers, predictors, registries, and configs.

This package holds featurizers, predictors, and component registries. Model
orchestration (factory, config IO/spec, zoo, composed training) lives under
`drevalpy.models`; this package re-exports those APIs for compatibility.
Legacy experiment workflows instantiate `~drevalpy.models.drp_model.DRPModel`
subclasses from `drevalpy.models`.
"""

from drevalpy.components.composed_model import ComposedModel
from drevalpy.components.config import (
    FeaturizerConfig,
    ModelConfig,
    PredictionMode,
    PredictorConfig,
)
from drevalpy.components.config_io import (
    model_config_from_dict,
    model_config_from_spec,
    model_config_from_yaml,
)
from drevalpy.components.extensions import (
    load_extension_dir,
    load_extension_file,
    load_extension_module,
    load_extensions,
)
from drevalpy.components.factory import (
    LEGACY_PREDICTOR_BY_MODEL_NAME,
    NAIVE_PREDICTOR_BY_MODEL_NAME,
    SKLEARN_PREDICTOR_BY_MODEL_NAME,
    legacy_model_config,
    model_config_for_name,
    naive_model_config,
    sklearn_model_config,
    sklearn_model_config_from_zoo,
)
from drevalpy.components.model_config_spec import build_model_config_from_spec
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
from drevalpy.models.zoo import (
    get_zoo_config,
    list_zoo_names,
    load_external_zoo_file,
    register_external_zoo_entry,
    zoo_model_config,
)

__all__ = [
    "ComponentDRPBridge",
    "ComposedModel",
    "FeaturizerConfig",
    "LEGACY_PREDICTOR_BY_MODEL_NAME",
    "ModelConfig",
    "NAIVE_PREDICTOR_BY_MODEL_NAME",
    "PredictorConfig",
    "PredictionMode",
    "SKLEARN_PREDICTOR_BY_MODEL_NAME",
    "build_model_config_from_spec",
    "ensure_components_registered",
    "format_model_id",
    "get_cell_line_featurizer",
    "get_drug_featurizer",
    "get_predictor",
    "get_zoo_config",
    "legacy_model_config",
    "list_cell_line_featurizer_metadata",
    "list_cell_line_featurizers",
    "list_drug_featurizer_metadata",
    "list_drug_featurizers",
    "list_predictor_metadata",
    "list_predictors",
    "list_zoo_names",
    "load_extension_dir",
    "load_extension_file",
    "load_extension_module",
    "load_extensions",
    "load_external_zoo_file",
    "model_config_for_name",
    "model_config_from_dict",
    "model_config_from_spec",
    "model_config_from_yaml",
    "naive_model_config",
    "parse_model_id",
    "preview_sklearn_estimator",
    "register_builtin_components",
    "register_cell_line_featurizer",
    "register_drug_featurizer",
    "register_external_zoo_entry",
    "register_predictor",
    "restore_naive_to_components",
    "restore_sklearn_to_components",
    "sklearn_model_config",
    "sklearn_model_config_from_zoo",
    "sync_naive_from_components",
    "sync_sklearn_from_components",
    "zoo_model_config",
]

_BRIDGE_LAZY_EXPORTS = frozenset(
    {
        "ComponentDRPBridge",
        "preview_sklearn_estimator",
        "restore_naive_to_components",
        "restore_sklearn_to_components",
        "sync_naive_from_components",
        "sync_sklearn_from_components",
    }
)


def __getattr__(name: str):
    if name in _BRIDGE_LAZY_EXPORTS:
        from drevalpy.models import _component_bridge

        return getattr(_component_bridge, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
