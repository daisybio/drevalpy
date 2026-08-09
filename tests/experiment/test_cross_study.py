"""Tests for experiment_randomization view building."""

from __future__ import annotations

from drevalpy.experiment.randomization import build_randomization_test_views

from drevalpy.models._model_lookup import get_model_class


def test_build_randomization_test_views_svrc() -> None:
    model_class = get_model_class("ElasticNet")
    views = build_randomization_test_views(model_class, ["SVRC"])
    assert views
    assert all(len(v) == 1 for v in views.values())
    assert all(key.startswith("SVRC_") for key in views)
