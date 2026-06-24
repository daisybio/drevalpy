"""Tests for package boundary between components and models."""

from __future__ import annotations

import importlib
import sys


def test_models_utils_reexports_shared_data_helpers() -> None:
    from drevalpy import models

    utils = importlib.import_module("drevalpy.models.utils")
    assert hasattr(utils, "load_and_select_gene_features")
    assert hasattr(utils, "ProteomicsMedianCenterAndImputeTransformer")
    assert models.MODEL_FACTORY["ElasticNet"] is not None


def test_native_component_registration_does_not_import_literature_models() -> None:
    from drevalpy.components.registry import clear_predictor_registry
    from drevalpy.components.registry.core import predictor_registry

    clear_predictor_registry()
    try:
        from drevalpy.components.register_builtins import register_native_components

        register_native_components()
        names = predictor_registry.list_names()
        assert "elasticNet" in names
        assert "naiveMean" in names
        assert "dipk" not in names
        assert "precily" not in names
    finally:
        clear_predictor_registry()
        from drevalpy.components.register_builtins import ensure_components_registered

        ensure_components_registered()


def test_component_featurizers_import_from_data_not_models_utils() -> None:
    view_module = sys.modules.get("drevalpy.components.featurizers.cell_line.view")
    if view_module is None:
        import drevalpy.components.featurizers.cell_line.view as view_module
    assert view_module is not None
    assert view_module.__file__ is not None
    source_path = view_module.__file__
    assert source_path is not None
    text = open(source_path, encoding="utf-8").read()
    assert "drevalpy.data.preprocessing" in text
    assert "drevalpy.models.utils" not in text


def test_component_predictors_import_from_data_not_models_utils() -> None:
    for module_name in (
        "drevalpy.components.predictors.baselines.sklearn_models",
        "drevalpy.components.predictors.baselines.naive_pred",
        "drevalpy.components.predictors.literature.impl.dipk.dipk",
        "drevalpy.components.predictors.literature.impl.simple_neural_network.utils",
    ):
        module = importlib.import_module(module_name)
        source_path = module.__file__
        assert source_path is not None
        text = open(source_path, encoding="utf-8").read()
        assert "drevalpy.models.utils" not in text
        assert "drevalpy.models.lightning_metrics_mixin" not in text


def test_models_lightning_metrics_mixin_reexports_component_mixin() -> None:
    from drevalpy.components.lightning_metrics_mixin import RegressionMetricsMixin as ComponentMixin
    from drevalpy.models.lightning_metrics_mixin import RegressionMetricsMixin as ModelsMixin

    assert ComponentMixin is ModelsMixin


def test_bridge_lives_in_models_layer() -> None:
    from drevalpy.components.drp_bridge import ComponentDRPBridge as ShimBridge
    from drevalpy.models._component_bridge import ComponentDRPBridge

    assert ComponentDRPBridge is ShimBridge


def test_orchestration_lives_in_models_layer() -> None:
    import drevalpy.components as components_pkg
    import drevalpy.components.factory as components_factory
    import drevalpy.models.composed_model as models_composed
    import drevalpy.models.config_io as models_config_io
    import drevalpy.models.factory as models_factory
    import drevalpy.models.model_config_spec as models_spec
    import drevalpy.models.zoo as models_zoo

    assert components_factory.model_config_for_name is models_factory.model_config_for_name
    assert components_pkg.get_zoo_config is models_zoo.get_zoo_config
    from drevalpy.components.composed_model import ComposedModel as ComponentsComposed

    assert ComponentsComposed is models_composed.ComposedModel
    assert models_config_io.model_config_from_yaml.__module__ == "drevalpy.models.config_io"
    assert models_spec.build_model_config_from_spec.__module__ == "drevalpy.models.model_config_spec"
