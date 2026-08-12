"""Smoke test mirror for srmf predictor package."""

from __future__ import annotations

import pytest

from drevalpy.components.predictors.literature.srmf.predictor import SRMFPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry._builtins import ensure_predictor_registered
from drevalpy.registry.predictor import get as get_predictor


def test_srmf_predictor_registry_name() -> None:
    ensure_predictor_registered("srmf")
    assert get_predictor("srmf") is SRMFPredictor


def test_structured_predictor_set_state_raises_on_invalid_blob() -> None:
    predictor = SRMFPredictor()
    with pytest.raises(PredictorStateError):
        predictor.set_state({"payload": b"invalid"})
