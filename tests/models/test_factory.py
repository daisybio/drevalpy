"""Tests for factory/zoo name resolution parity with configure flat HP."""

from __future__ import annotations

from drevalpy.components.tuning.drp_hyperparameters import config_from_public_hyperparameters
from drevalpy.models import construct_model
from drevalpy.models.factory import model_config_for_name


def test_model_config_for_name_matches_configure_path_for_views() -> None:
    model_cls = construct_model("MultiViewRandomForest")
    flat = {"cell_line_views": ["gene_expression"], "n_estimators": 8}
    via_factory = model_config_for_name("MultiViewRandomForest", flat)
    via_configure = config_from_public_hyperparameters(model_cls, flat)
    assert via_configure is not None
    assert via_factory.cell_line_featurizer is not None
    assert via_configure.cell_line_featurizer is not None
    assert via_factory.cell_line_featurizer.name == via_configure.cell_line_featurizer.name
    assert via_factory.predictor.hyperparameters["n_estimators"] == 8
    assert "cell_line_views" not in via_factory.predictor.hyperparameters
