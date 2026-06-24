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
    from drevalpy.models._component_bridge import ComponentDRPBridge
    from drevalpy.components.drp_bridge import ComponentDRPBridge as ShimBridge

    assert ComponentDRPBridge is ShimBridge
