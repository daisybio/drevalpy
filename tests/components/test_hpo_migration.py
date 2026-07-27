"""Guard tests for the structured HPO migration."""

from __future__ import annotations

import inspect


def test_experiment_tuning_does_not_use_parameter_grid() -> None:
    from drevalpy import experiment

    source = inspect.getsource(experiment.hpam_tune)
    assert "ParameterGrid" not in source
    assert "grid_search" not in source


def test_drp_model_does_not_load_yaml_hyperparameters() -> None:
    from drevalpy.models import drp_model

    source = inspect.getsource(drp_model.DRPModel.get_hyperparameter_set)
    assert "yaml" not in source.lower()
    assert "ParameterGrid" not in source


def test_legacy_baseline_hyperparameters_yaml_removed() -> None:
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "drevalpy"
        / "components"
        / "predictors"
        / "baselines"
        / "hyperparameters.yaml"
    )
    assert not path.exists()
