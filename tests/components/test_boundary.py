"""Tests for package boundary between components and models."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_native_component_registration_does_not_import_literature_models() -> None:
    from drevalpy.components.registry.predictor_registry import predictor_registry

    predictor_registry.clear()
    try:
        from drevalpy.components.core.plugins.register_builtins import register_native_components

        register_native_components()
        names = predictor_registry.list_names()
        assert "elasticNet" in names
        assert "naiveMean" in names
        assert "dipk" not in names
        assert "precily" not in names
    finally:
        predictor_registry.clear()
        from drevalpy.components.core.plugins.register_builtins import register_builtin_components

        register_builtin_components()


def test_component_featurizers_import_from_features_not_models_utils() -> None:
    for module_name in (
        "drevalpy.components.featurizers.cell_line.scaled_gene_expression",
        "drevalpy.components.featurizers.cell_line.normalized_proteomics",
    ):
        module = importlib.import_module(module_name)
        source_path = module.__file__
        assert source_path is not None
        text = Path(source_path).read_text(encoding="utf-8")
        assert "drevalpy.components.core.features.preprocessing" in text
        assert "drevalpy.models.utils" not in text


def test_component_predictors_avoid_models_utils() -> None:
    for module_name in (
        "drevalpy.components.predictors.sklearn_models",
        "drevalpy.components.predictors.naive",
        "drevalpy.components.predictors.literature.dipk.predictor",
        "drevalpy.components.predictors.neural_network.network",
    ):
        module = importlib.import_module(module_name)
        source_path = module.__file__
        assert source_path is not None
        text = Path(source_path).read_text(encoding="utf-8")
        assert "drevalpy.models.utils" not in text
        assert "drevalpy.models.lightning_metrics_mixin" not in text


def test_models_lightning_metrics_mixin_reexports_component_mixin() -> None:
    from drevalpy.components.core.utils.lightning_metrics_mixin import RegressionMetricsMixin as ComponentMixin
    from drevalpy.models.lightning_metrics_mixin import RegressionMetricsMixin as ModelsMixin

    assert ComponentMixin is ModelsMixin


def test_orchestration_lives_in_models_layer() -> None:
    import drevalpy.components as components_pkg
    import drevalpy.models._component_stack as component_stack
    import drevalpy.models.config.io as models_config_io
    import drevalpy.models.config.spec as models_spec
    import drevalpy.models.factory as models_factory
    import drevalpy.models.zoo as models_zoo

    assert not hasattr(components_pkg, "ComposedModel")
    assert not hasattr(components_pkg, "model_config_for_name")
    assert not hasattr(components_pkg, "get_zoo_config")
    assert models_factory.model_config_for_name.__module__ == "drevalpy.models.factory"
    assert component_stack.build_component_stack.__module__ == "drevalpy.models._component_stack"
    assert models_config_io.from_yaml.__module__ == "drevalpy.models.config.io"
    assert models_config_io.from_spec.__module__ == "drevalpy.models.config.io"
    assert models_spec.zoo_config.__module__ == "drevalpy.models.config.spec"
    assert models_zoo.get_zoo_config.__module__ == "drevalpy.models.zoo"
