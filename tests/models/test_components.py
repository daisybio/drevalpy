"""Tests for fresh component construction behind the model pipeline."""

from __future__ import annotations

from drevalpy.models._components import build_components
from drevalpy.models.config import from_spec
from drevalpy.models.tuning.search_space import resolve_model_config


def test_build_components_creates_predictor_and_configured_featurizers() -> None:
    resolved = resolve_model_config(from_spec("ElasticNet"))
    sample, drug, predictor = build_components(resolved)

    assert sample is not None
    assert drug is not None
    assert predictor.__class__.__name__


def test_build_components_returns_independent_instances() -> None:
    resolved = resolve_model_config(from_spec("ElasticNet"))
    first = build_components(resolved)
    second = build_components(resolved)

    assert first[0] is not second[0]
    assert first[1] is not second[1]
    assert first[2] is not second[2]
