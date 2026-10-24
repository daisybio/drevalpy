"""Tests for hpo migration."""

from __future__ import annotations

import inspect
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "drevalpy"


def test_removed_legacy_hpo_artifacts_stay_removed() -> None:
    from drevalpy.models import drp_model
    from drevalpy.models.tuning.hpo import hpam_tune

    assert "ParameterGrid" not in inspect.getsource(hpam_tune)
    assert "yaml" not in inspect.getsource(drp_model.DRPModel.__init__).lower()
    literature = SRC / "components" / "predictors" / "literature"
    assert not (literature / "impl").exists()
    assert not list(literature.glob("*_predictor.py"))
    assert not list(SRC.rglob("hyperparameters.yaml"))
